from typing import List, Tuple
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def neighbors(self, entity_name: str) -> List[Tuple[str, str]]:
        query = """
        MATCH (e:Entity)-[r]->(n)
        WHERE e.name = $entity_name
        RETURN type(r) as rel, n.name as neighbor
        """
        with self.driver.session() as session:
            res = session.run(query, entity_name=entity_name)
            return [(r['rel'], r['neighbor']) for r in res]
