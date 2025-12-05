from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import socket
from urllib.parse import urlparse
from neo4j import GraphDatabase, __version__ as neo4j_version
from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError
from .settings import Neo4jConfig
from .logger import Logger


class Repository(ABC):
    """리포지토리 인터페이스 (DIP - 의존성 역전)"""
    
    @abstractmethod
    def clear_data(self, clear_type: str, identifiers: List[str] = None):
        """데이터 정리"""
        pass
    
    @abstractmethod
    def execute_queries(self, queries: List[str]):
        """쿼리 실행"""
        pass
    
    @abstractmethod
    def close(self):
        """연결 종료"""
        pass


class Neo4jRepository(Repository):
    """Neo4j 데이터 접근 클래스 (SRP)"""
    
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver = None
        self.logger = Logger.get_logger(self.__class__.__name__)
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Neo4j 연결 초기화 및 상세 진단"""
        self.logger.info("=== Neo4j 연결 초기화 시작 ===")
        self.logger.info(f"Neo4j Python Driver 버전: {neo4j_version}")
        self.logger.info(f"연결 URI: {self.config.uri}")
        self.logger.info(f"사용자명: {self.config.username}")
        self.logger.info(f"데이터베이스: {self.config.database}")
        
        # 1. URI 파싱 및 검증
        try:
            parsed_uri = urlparse(self.config.uri)
            self.logger.info(f"파싱된 URI - 스키마: {parsed_uri.scheme}, 호스트: {parsed_uri.hostname}, 포트: {parsed_uri.port}")
            
            if not parsed_uri.hostname:
                raise ValueError(f"잘못된 URI 형식: {self.config.uri}")
                
        except Exception as e:
            self.logger.error(f"URI 파싱 실패: {e}")
            raise
        
        # 2. 네트워크 연결 테스트 (선택적)
        if parsed_uri.hostname and parsed_uri.port:
            self._test_network_connectivity(parsed_uri.hostname, parsed_uri.port)
        
        # 3. Neo4j 드라이버 생성
        try:
            self.logger.info("Neo4j 드라이버 생성 중...")
            
            # 연결 옵션 설정
            driver_config = {
                "max_connection_lifetime": 3600,  # 1시간
                "max_connection_pool_size": 50,
                "connection_acquisition_timeout": 60,  # 60초
                "connection_timeout": 30,  # 30초
                "keep_alive": True,
                "user_agent": "airflow-bpmn-loader/1.0"
            }
            
            self.logger.info(f"드라이버 설정: {driver_config}")
            
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                **driver_config
            )
            
            self.logger.info("Neo4j 드라이버 생성 성공")
            
        except Exception as e:
            self.logger.error(f"Neo4j 드라이버 생성 실패: {e}")
            raise
        
        # 4. 연결 테스트
        self._test_connection()
    
    def _test_network_connectivity(self, hostname: str, port: int):
        """네트워크 연결 테스트"""
        self.logger.info(f"=== 네트워크 연결 테스트: {hostname}:{port} ===")
        
        try:
            # DNS 해석 테스트
            self.logger.info("DNS 해석 테스트 중...")
            ip_addresses = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
            resolved_ips = [addr[4][0] for addr in ip_addresses]
            self.logger.info(f"DNS 해석 성공: {hostname} -> {resolved_ips}")
            
            # TCP 연결 테스트
            self.logger.info("TCP 연결 테스트 중...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10초 타임아웃
            
            start_time = time.time()
            result = sock.connect_ex((hostname, port))
            connection_time = time.time() - start_time
            
            sock.close()
            
            if result == 0:
                self.logger.info(f"TCP 연결 성공 ({connection_time:.2f}초)")
            else:
                self.logger.error(f"TCP 연결 실패: 에러 코드 {result}")
                
        except socket.gaierror as e:
            self.logger.error(f"DNS 해석 실패: {e}")
        except Exception as e:
            self.logger.error(f"네트워크 테스트 실패: {e}")
    
    def _test_connection(self):
        """Neo4j 연결 및 인증 테스트"""
        self.logger.info("=== Neo4j 연결 테스트 ===")
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"연결 테스트 시도 {attempt}/{max_retries}")
                
                # 드라이버 검증
                self.driver.verify_connectivity()
                self.logger.info("드라이버 연결성 검증 성공")
                
                # 실제 쿼리 테스트
                with self.driver.session(database=self.config.database) as session:
                    start_time = time.time()
                    result = session.run("RETURN 1 as test").single()
                    query_time = time.time() - start_time
                    
                    if result and result["test"] == 1:
                        self.logger.info(f"테스트 쿼리 성공 ({query_time:.3f}초)")
                        
                        # 데이터베이스 정보 조회
                        try:
                            db_info = session.run("CALL db.info()").single()
                            if db_info:
                                self.logger.info(f"데이터베이스 정보: {dict(db_info)}")
                        except Exception as e:
                            self.logger.warning(f"데이터베이스 정보 조회 실패 (권한 부족일 수 있음): {e}")
                        
                        self.logger.info("=== Neo4j 연결 테스트 완료 ===")
                        return
                    else:
                        raise RuntimeError("테스트 쿼리 결과가 예상과 다름")
                        
            except AuthError as e:
                self.logger.error(f"인증 실패 (시도 {attempt}): {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Neo4j 인증 실패: 사용자명/비밀번호를 확인하세요. {e}")
                
            except ServiceUnavailable as e:
                self.logger.error(f"서비스 이용 불가 (시도 {attempt}): {e}")
                if attempt < max_retries:
                    wait_time = attempt * 2
                    self.logger.info(f"{wait_time}초 대기 후 재시도...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Neo4j 서비스에 연결할 수 없습니다: {e}")
                
            except ClientError as e:
                self.logger.error(f"클라이언트 오류 (시도 {attempt}): {e}")
                if "database does not exist" in str(e).lower():
                    raise RuntimeError(f"데이터베이스 '{self.config.database}'가 존재하지 않습니다")
                elif attempt == max_retries:
                    raise RuntimeError(f"Neo4j 클라이언트 오류: {e}")
                
            except Exception as e:
                self.logger.error(f"예상치 못한 오류 (시도 {attempt}): {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Neo4j 연결 테스트 실패: {e}")
                else:
                    time.sleep(2)
    
    def clear_data(self, clear_type: str, identifiers: List[str] = None):
        """
        데이터 정리
        clear_type: "all" | "container" | "model" | "none"
        identifiers: container_id 또는 model_keys 리스트
        """
        if clear_type == "none":
            self.logger.info("데이터 정리 없음 (clear_type=none)")
            return
        
        self.logger.info(f"=== 데이터 정리 시작: {clear_type} ===")
        
        try:
            with self.driver.session(database=self.config.database) as session:
                if clear_type == "all":
                    self.logger.info("모든 데이터 삭제 중...")
                    session.run("MATCH (n) DETACH DELETE n")
                    self.logger.info("모든 데이터 삭제 완료")
                    
                elif clear_type == "container" and identifiers:
                    container_id = identifiers[0]
                    self.logger.info(f"컨테이너 '{container_id}' 데이터 삭제 중...")
                    
                    # 관계부터 삭제
                    result1 = session.run("MATCH ()-[r]-() WHERE r.containerId=$cid DELETE r", cid=container_id)
                    self.logger.info(f"관계 삭제 완료: {result1.consume().counters}")
                    
                    # 컨테이너 하위 노드 삭제 (컨테이너 노드 자체는 유지)
                    result2 = session.run("""
                        MATCH (n)
                        WHERE n.containerId=$cid AND NOT (n:Project OR n:Question)
                        DETACH DELETE n
                    """, cid=container_id)
                    self.logger.info(f"노드 삭제 완료: {result2.consume().counters}")
                    
                elif clear_type == "model" and identifiers:
                    model_keys = identifiers
                    self.logger.info(f"모델 {len(model_keys)}개 삭제 중...")
                    
                    result1 = session.run("MATCH ()-[r]-() WHERE r.modelKey IN $mks DELETE r", mks=model_keys)
                    self.logger.info(f"관계 삭제 완료: {result1.consume().counters}")
                    
                    result2 = session.run("MATCH (n) WHERE n.modelKey IN $mks DETACH DELETE n", mks=model_keys)
                    self.logger.info(f"노드 삭제 완료: {result2.consume().counters}")
                    
        except Exception as e:
            self.logger.error(f"데이터 정리 실패: {e}")
            raise
    
    def execute_queries(self, queries: List[str]):
        """쿼리 배치 실행"""
        if not queries:
            self.logger.warning("실행할 쿼리가 없습니다")
            return
        
        self.logger.info(f"=== 쿼리 배치 실행 시작: {len(queries)}개 ===")
        
        start_time = time.time()
        successful_queries = 0
        failed_queries = 0
        
        try:
            with self.driver.session(database=self.config.database) as session:
                for i, query in enumerate(queries, 1):
                    try:
                        query_start = time.time()
                        result = session.run(query)
                        counters = result.consume().counters
                        query_time = time.time() - query_start
                        
                        successful_queries += 1
                        
                        # 상세 로깅 (처음 5개와 마지막 5개만)
                        if i <= 5 or i > len(queries) - 5:
                            self.logger.info(f"쿼리 {i}: {query_time:.3f}초, 카운터: {counters}")
                        
                        # 진행률 로깅
                        if i % 50 == 0 or i == len(queries):
                            elapsed_time = time.time() - start_time
                            avg_time = elapsed_time / i
                            estimated_remaining = (len(queries) - i) * avg_time
                            
                            self.logger.info(
                                f"진행률: {i}/{len(queries)} ({i/len(queries)*100:.1f}%) "
                                f"- 성공: {successful_queries}, 실패: {failed_queries} "
                                f"- 평균 속도: {avg_time:.3f}초/쿼리 "
                                f"- 예상 남은 시간: {estimated_remaining:.1f}초"
                            )
                            
                    except Exception as e:
                        failed_queries += 1
                        self.logger.error(f"쿼리 실행 오류 [{i}]: {e}")
                        self.logger.error(f"문제 쿼리: {query[:200]}...")
                        
                        # 연결 문제인 경우 즉시 중단
                        if "connection" in str(e).lower() or "unavailable" in str(e).lower():
                            self.logger.error("연결 문제로 인한 실행 중단")
                            raise
                        
                        # 계속 진행할지 결정 (실패율이 높으면 중단)
                        if failed_queries > successful_queries and i > 10:
                            self.logger.error("실패율이 높아 실행 중단")
                            raise RuntimeError(f"쿼리 실패율이 너무 높습니다: {failed_queries}/{i}")
            
            total_time = time.time() - start_time
            self.logger.info(
                f"쿼리 배치 실행 완료: "
                f"총 {len(queries)}개 중 성공 {successful_queries}개, 실패 {failed_queries}개 "
                f"({total_time:.2f}초, 평균 {total_time/len(queries):.3f}초/쿼리)"
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(
                f"쿼리 배치 실행 실패: {e} "
                f"(진행률: {successful_queries + failed_queries}/{len(queries)}, "
                f"경과시간: {total_time:.2f}초)"
            )
            raise
    
    def execute_single_query(self, query: str, parameters: Dict[str, Any] = None) -> Any:
        """단일 쿼리 실행"""
        self.logger.debug(f"단일 쿼리 실행: {query[:100]}...")
        
        try:
            start_time = time.time()
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, parameters or {})
                data = result.data()
                query_time = time.time() - start_time
                
                self.logger.debug(f"쿼리 완료: {len(data)}개 레코드, {query_time:.3f}초")
                return data
                
        except Exception as e:
            self.logger.error(f"단일 쿼리 실행 실패: {e}")
            self.logger.error(f"쿼리: {query}")
            self.logger.error(f"파라미터: {parameters}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """연결 정보 조회"""
        try:
            with self.driver.session(database=self.config.database) as session:
                # 서버 정보
                server_info = session.run("CALL dbms.components()").data()
                
                # 현재 사용자
                current_user = session.run("CALL dbms.showCurrentUser()").single()
                
                # 데이터베이스 목록 (권한이 있는 경우)
                try:
                    databases = session.run("SHOW DATABASES").data()
                except:
                    databases = []
                
                return {
                    'server_info': server_info,
                    'current_user': dict(current_user) if current_user else None,
                    'databases': databases,
                    'connected_database': self.config.database
                }
        except Exception as e:
            self.logger.error(f"연결 정보 조회 실패: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Neo4j 연결 종료"""
        if self.driver:
            try:
                self.driver.close()
                self.logger.info("Neo4j 연결 종료 완료")
            except Exception as e:
                self.logger.error(f"Neo4j 연결 종료 중 오류: {e}")
            finally:
                self.driver = None


class CypherBuilder:
    """Cypher 쿼리 생성 유틸리티 (SRP)"""
    
    @staticmethod
    def escape_string(text: str) -> str:
        """문자열 이스케이프 처리"""
        if not text:
            return ""
        return (text
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r"))
    
    @staticmethod
    def format_properties(props: Dict[str, Any], alias: str = "n") -> str:
        """속성을 Cypher 형식으로 변환"""
        if not props:
            return f"{alias}.dummy = null"
        
        parts = []
        for k, v in props.items():
            if isinstance(v, bool):
                parts.append(f"{alias}.{k} = {str(v).lower()}")
            elif isinstance(v, (int, float)):
                parts.append(f"{alias}.{k} = {v}")
            else:
                escaped = CypherBuilder.escape_string(str(v))
                parts.append(f"{alias}.{k} = '{escaped}'")
        return ", ".join(parts)
    
    @staticmethod
    def create_node_query(node_data: Dict[str, Any]) -> str:
        """노드 생성 쿼리"""
        node_type = node_data['type']
        node_id = CypherBuilder.escape_string(node_data['id'])
        node_name = CypherBuilder.escape_string(node_data.get('name', '') or node_data['id'])
        props = node_data.get('properties', {})
        props_str = CypherBuilder.format_properties(props, "n")
        
        return f"""
MERGE (n:{node_type} {{id:'{node_id}'}})
SET n.name = '{node_name}',
    {props_str}
""".strip()
    
    @staticmethod
    def create_relationship_query(rel_data: Dict[str, Any]) -> str:
        """관계 생성 쿼리"""
        source = CypherBuilder.escape_string(rel_data['source'])
        target = CypherBuilder.escape_string(rel_data['target'])
        rel_type = rel_data['type']
        props = rel_data.get('properties', {})
        props_str = CypherBuilder.format_properties(props, "r")
        
        return f"""
MATCH (a {{id:'{source}'}}), (b {{id:'{target}'}})
CREATE (a)-[r:{rel_type}]->(b)
SET {props_str}
""".strip()