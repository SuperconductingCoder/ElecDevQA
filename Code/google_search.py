from googleapiclient.discovery import build
import json
def Google(Search_term, api_key, search_engine_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)

    res = service.cse().list(
        q = Search_term,
        cx = search_engine_id,
        **kwargs
    ).execute()
    
    try:
        return res['items']
    except:
        return False
if __name__ == "__main__":
    search_results = Google(question, api_key, search_engine_id, num=100)