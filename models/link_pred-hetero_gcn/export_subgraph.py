### Code to export subgraph from neo4j
## please note files go into import directory in neo4j
## if you need to use them move to another folder

##################################################################################
# CONNECT TO NEO4J COMPTOXAI
##################################################################################
from comptox_ai.db.graph_db import GraphDB
db = GraphDB()
##################################################################################

##################################################################################
# GET NODES
##################################################################################
# chemicals
db.run_cypher("""
WITH "MATCH (n:Chemical) WHERE n.maccs IS NOT NULL RETURN id(n) AS node, n.maccs AS maccs;" AS chemicalsquery
CALL apoc.export.csv.query(chemicalsquery, "chemicals.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
""")

# assays              
db.run_cypher("""
WITH "MATCH (n:Assay) RETURN id(n) AS node;" AS assaysquery
CALL apoc.export.csv.query(assaysquery, "assays.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
""")
              
# genes             
db.run_cypher("""
WITH "MATCH (n:Gene) RETURN id(n) AS node;" AS genesquery
CALL apoc.export.csv.query(genesquery, "genes.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
""")
              
# diseases             
db.run_cypher("""
WITH "MATCH (n:Disease) RETURN id(n) AS node;" AS diseasesquery
CALL apoc.export.csv.query(diseasesquery, "diseases.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
""")
##################################################################################

##################################################################################
# GET RELATIONSHIPS
##################################################################################
# chemical-assay
db.run_cypher("""
WITH "MATCH (n:Chemical)-[r]-(m:Assay) WHERE n.maccs IS NOT NULL RETURN id(n) AS node1, TYPE(r) AS edge, id(m) AS node2;" AS edgeqry
CALL apoc.export.csv.query(edgeqry, "chemical-assay.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
""")
              
# chemical-gene
db.run_cypher("""
WITH "MATCH (n:Chemical)-[r]-(m:Gene) WHERE n.maccs IS NOT NULL RETURN id(n) AS node1, TYPE(r) AS edge, id(m) AS node2;" AS edgeqry
CALL apoc.export.csv.query(edgeqry, "chemical-gene.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
""")
              
# gene-gene
db.run_cypher("""
WITH "MATCH (n:Gene)-[r]-(m:Gene) RETURN id(n) AS node1, TYPE(r) AS edge, id(m) AS node2;" AS edgeqry
CALL apoc.export.csv.query(edgeqry, "gene-gene.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
""")

# chemical-disease
db.run_cypher("""
WITH "MATCH (n:Chemical)-[r]-(m:Disease) WHERE n.maccs IS NOT NULL RETURN id(n) AS node1, TYPE(r) AS edge, id(m) AS node2;" AS edgeqry
CALL apoc.export.csv.query(edgeqry, "chemical-disease.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
""")
##################################################################################