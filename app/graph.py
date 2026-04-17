from neo4j import GraphDatabase

# 🔹 Connect to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")  # change password if needed
)


def add_profile(profile_text):
    try:
        with driver.session() as session:
            session.run(
                "MERGE (p:Profile {text: $text})",
                text=profile_text
            )
    except Exception as e:
        print("Graph error:", e)
