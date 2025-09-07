from neo4j import GraphDatabase

def write_SemiDevKG_jsonl(driver, file):
    file_path  = f"\"file:///{file}\""
    load_file_query = f'''CALL apoc.load.csv({file_path}) YIELD value '''
    
    with driver.session() as session:
        result = session.run(load_file_query) 

def create_index(driver):
    
    queries = [
        '''
        CREATE FULLTEXT INDEX fulltext_device_name IF NOT EXISTS FOR (n:Device) ON EACH [n.name];
        ''',
        '''
        CREATE FULLTEXT INDEX fulltext_system_name IF NOT EXISTS FOR (n:System) ON EACH [n.name];
        ''',
        '''
        CREATE FULLTEXT INDEX fulltext_technique_name IF NOT EXISTS FOR (n:Technique) ON EACH [n.technique];''',
        '''
        CREATE FULLTEXT INDEX fulltext_purpose_name IF NOT EXISTS FOR (n:Purpose) ON EACH [n.purpose];
        ''',
        '''
        CREATE FULLTEXT INDEX fulltext_component_name IF NOT EXISTS FOR (n:Component) ON EACH [n.name];
        ''',
        '''
        CREATE FULLTEXT INDEX fulltext_property_name IF NOT EXISTS FOR (n:Property) ON EACH [n.name];
        ''',
        '''
        CREATE FULLTEXT INDEX fulltext_material_name IF NOT EXISTS FOR (n:Material) ON EACH [n.name];
        ''',
        '''
        CREATE FULLTEXT INDEX fulltext_function_name IF NOT EXISTS FOR (n:Function) ON EACH [n.function];''',
        '''
        CREATE FULLTEXT INDEX fulltext_feature_name IF NOT EXISTS FOR (n:Feature) ON EACH [n.feature];
        ''',
        '''
        CREATE FULLTEXT INDEX fulltext_genre_name IF NOT EXISTS FOR (n:Genre) ON EACH [n.name];''']
        
    for query in queries:
        print(query)
        with driver.session() as session:
            session.run(query)
            print("---------Fulltext index created successfully.------")


if __name__ == "__main__":
    URI = 'bolt://localhost:7687'
    usr = 'neo4j'
    password = '******'
    my_driver = GraphDatabase.driver(URI, auth=(usr, password))
    
    file = "SemiDevKG.csv"
    write_SemiDevKG_jsonl(my_driver, file)
    try:
        create_index(my_driver)
    except:
        pass
    