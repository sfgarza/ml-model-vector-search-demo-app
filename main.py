from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from dotenv import dotenv_values

config = dotenv_values(".env")

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
# In this example we are using a pre-trained sentence transformer model that creates a vector embedding that maps to 109 different languages.
# The same general concept can be applied to any model that generates a vector embedding in order to perform a similiarity search.
language_agnostic_search_model = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')

# Initialize Elasticsearch client
es = Elasticsearch(
    [config['ES_HOST']],
    basic_auth=(config['ES_USER'], config['ES_PASS']),
    #ca_certs='./http_ca.crt',
    verify_certs=False,
)
# Check Elasticsearch connection
if es.ping():
    print("Elasticsearch is connected")
else:
    raise ValueError("Elasticsearch connection failed")

# Define index name and mapping
index_name = 'products'
mapping = {
    "mappings": {
        "properties": {
            # This is our vector embedding field, which we we will be using to perform a vector search on.
            "language-agnostic-search-embedding": {"type": "dense_vector", "dims": 768},
            # The rest are regular document properties which we use to display the search results.
            "product_id": {"type": "integer"},
            "spin": {"type": "keyword"},
            "product_title": {"type": "text"},
            "clean_product_description": {"type": "text"},
            "category_title": {"type": "keyword"},
            "category_description": {"type": "text"},
            "custom_category_text": {"type": "keyword"},
            "parent_title": {"type": "keyword"},
            "product_tags": {"type": "keyword"},
            "product_configurations": {
                "type": "nested",
                "properties": {
                    "product_configuration_url": {"type": "keyword"},
                    "product_configuration_id": {"type": "integer"},
                    "product_configuration_display_name": {"type": "keyword"},
                    "product_configuration_total_price": {"type": "float"},
                    "product_pictures": {
                        "type": "nested",
                        "properties": {
                            "product_picture_url": {"type": "keyword"},
                            "product_picture_id": {"type": "integer"},
                            "picture_entity_id": {"type": "integer"},
                            "priority": {"type": "integer"},
                            "title": {"type": "text"},
                            "description": {"type": "text"},
                            "picture_id": {"type": "integer"}
                        }
                    }
                }
            }
        }
    }
}

# Create the index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)

# Function to create a combined text field for gneratating vector embeddings.
#
# In this example we combine all of the text fields to "query all fields", however in production you may want
# to create separate vector mappings for each property in order to add weights & rank results by field weight.
def create_combined_text(document):
    combined_text = f"{document.get('product_title', '')} {document.get('clean_product_description', '')} " \
                    f"{document.get('category_title', '')} {document.get('category_description', '')} " \
                    f"{document.get('parent_title', '')} {document.get('custom_category_text', '')} " \
                    f"{' '.join(document.get('product_tags', []) if document.get('product_tags') else [])} " \
                    f"{' '.join(pc['product_configuration_display_name'] for pc in document.get('product_configurations', []))}"
    return combined_text


# Serves the search page.
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Indexing endpoint
@app.route('/index', methods=['POST'])
def index_document():
    data = request.json
    combined_text = create_combined_text(data)
    language_agnostic_embedding = language_agnostic_search_model.encode([combined_text])[0].tolist()

    doc_body = {
        "product_id": data.get('product_id', ''),
        "spin": data.get('spin', ''),
        "product_title": data.get('product_title', ''),
        "clean_product_description": data.get('clean_product_description', ''),
        "category_title": data.get('category_title', ''),
        "category_description": data.get('category_description', ''),
        "custom_category_text": data.get('custom_category_text', ''),
        "parent_title": data.get('parent_title', ''),
        "product_tags": data.get('product_tags', []) if data.get('product_tags') else [],
        "language-agnostic-search-embedding": language_agnostic_embedding,
        "product_configurations": data.get('product_configurations', [])
    }
    res = es.index(index=index_name, body=doc_body)
    return jsonify({"result": res["result"], "id": res["_id"]})

# Searching endpoint
@app.route('/search', methods=['POST'])
def search_document():
    data = request.json
    query = data['query']
    query_embedding = language_agnostic_search_model.encode([query])[0]

    # Here we define the search query, we are only performing cosineSimilarity search on the vector embeddings
    # but the query can be combined with keyword search in order to create a hybrid search.
    # https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#_combine_approximate_knn_with_other_features
    search_query = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'language-agnostic-search-embedding') + 1.0",
                "params": {"query_vector": query_embedding}
            }
        }
    }

    response = es.search(index=index_name, body={"query": search_query}, size=40)
    hits = response['hits']['hits']

    results = [{"score": hit['_score'], "product_title": hit['_source']['product_title'],
                "clean_product_description": hit['_source']['clean_product_description'],
                "category_title": hit['_source']['category_title'],
                "parent_title": hit['_source']['parent_title'],
                "product_configurations": hit['_source']['product_configurations'],
                "spin": hit['_source']['spin'],
                "category_description": hit['_source']['category_description'],
                "product_tags": hit['_source']['product_tags']} for hit in hits]
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config['PORT'])
