from dataclasses import dataclass
from typing import Optional

@dataclass
class Neo4jConfig:
    """Neo4j 연결 설정"""
    uri: str = ""
    username: str = ""
    password: str = ""
    database: str = "neo4j"


@dataclass
class OpenAIConfig:
    """OpenAI API 설정"""
    api_key: str = "" # 기본값 제공
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    translation_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens_full: int = 600                # tokens reserved for FULL output
    max_tokens_summary: int = 200             # tokens reserved for SUMMARY output
    timeout: int = 30                         # request timeout in seconds


class Settings:
    """설정 관리 클래스 (SRP)"""
    
    def __init__(self):
        self._neo4j_config: Optional[Neo4jConfig] = None
        self._openai_config: Optional[OpenAIConfig] = None

    @property
    def neo4j(self) -> Neo4jConfig:
        """Neo4j 설정 반환"""
        if self._neo4j_config is None:
            self._neo4j_config = Neo4jConfig()
        return self._neo4j_config
    
    @property
    def openai(self) -> OpenAIConfig:
        """OpenAI 설정 반환"""
        if self._openai_config is None:
            self._openai_config = OpenAIConfig()
        return self._openai_config

    def set_neo4j_config(self, uri: str, user_name: str, password: str, database: str):
        """컨테이너 설정 업데이트"""
        self._neo4j_config = Neo4jConfig(
            uri=uri,
            username=user_name,
            password=password,
            database=database
        )
    
    def set_openai_config(self, api_key: str, embedding_model: str = None, translation_model: str = None):
        """OpenAI 설정 업데이트"""
        config = self.openai  # 기존 설정 로드
        self._openai_config = OpenAIConfig(
            api_key=api_key,
            embedding_model=embedding_model or config.embedding_model,
            embedding_dimension=config.embedding_dimension,
            translation_model=translation_model or config.translation_model,
            temperature=config.temperature
        )