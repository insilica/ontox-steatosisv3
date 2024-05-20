import sysrev as sr, dotenv, os, sqlite3, pandas as pd, json

dotenv.load_dotenv()

project_id = 123619
client = sr.Client(os.getenv('SR_ADMIN_TOKEN'))
client.sync(project_id)
