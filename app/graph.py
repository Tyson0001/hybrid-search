from neo4j import GraphDatabase

# 🔹 Connection (change password if needed)
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# 🔹 Simple skill extraction
def extract_skills(text):
    words = text.lower().split()

    stopwords = {
        "and", "the", "with", "for", "of", "in", "to",
        "a", "an", "on", "is", "are"
    }

    # basic filtering
    skills = [
        w.strip(",.()")
        for w in words
        if w not in stopwords and len(w) > 3
    ]

    # unique + limit
    return list(set(skills[:10]))


# 🔹 Add profile + relationships
def add_profile(profile_text):
    try:
        with driver.session() as session:

            # ✅ Create Profile node
            session.run(
                "MERGE (p:Profile {text: $text})",
                text=profile_text
            )

            skills = extract_skills(profile_text)

            # ✅ Link Profile → Skills
            for skill in skills:
                session.run(
                    """
                    MERGE (s:Skill {name: $skill})
                    WITH s
                    MATCH (p:Profile {text: $text})
                    MERGE (p)-[:HAS_SKILL]->(s)
                    """,
                    skill=skill,
                    text=profile_text
                )

            # ✅ Create relationships between skills
            for i in range(len(skills) - 1):
                session.run(
                    """
                    MATCH (a:Skill {name: $s1})
                    MATCH (b:Skill {name: $s2})
                    MERGE (a)-[:RELATED_TO]->(b)
                    """,
                    s1=skills[i],
                    s2=skills[i + 1]
                )

    except Exception as e:
        print("Graph error:", e)


# 🔹 Get related skills (Traversal)
def get_related_skills(skill):
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (s:Skill {name:$skill})-[:RELATED_TO]->(other)
                RETURN other.name AS skill
                LIMIT 5
                """,
                skill=skill.lower()
            )

            return [r["skill"] for r in result]

    except Exception as e:
        print("Graph error:", e)
        return []


# 🔹 Get profiles by skill (Traversal)
def get_profiles_by_skill(skill):
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (p:Profile)-[:HAS_SKILL]->(s:Skill {name:$skill})
                RETURN p.text AS profile
                LIMIT 5
                """,
                skill=skill.lower()
            )

            return [r["profile"] for r in result]

    except Exception as e:
        print("Graph error:", e)
        return []


# 🔹 Close connection (optional)
def close():
    driver.close()
