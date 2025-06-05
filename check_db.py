from src.bistoury.database import DatabaseManager
from src.bistoury.config import Config

config = Config.load_from_env()
db = DatabaseManager(config)

print('📊 Database counts:')
tables = ['trades', 'candles_1m', 'orderbook_snapshots']
for table in tables:
    try:
        count = db.execute(f'SELECT COUNT(*) FROM {table}')[0][0]
        print(f'  {table}: {count:,} records')
    except Exception as e:
        print(f'  {table}: Error - {e}') 