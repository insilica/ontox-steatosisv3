import sysrev as sr, dotenv, os, sqlite3, pandas as pd, json

dotenv.load_dotenv()

project_id = 123619
client = sr.Client(os.getenv('SR_ADMIN_TOKEN'))
client.sync(project_id)

SR_ADMIN_TOKEN="bc46c94cf6c55ccc48cf46c2"
OPENAI_API_KEY='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'