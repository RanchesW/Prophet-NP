import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Configure Oracle client
oracle_client_path = r"C:\Users\aalik\Documents\instantclient_23_4"
if not os.path.exists(oracle_client_path):
    raise Exception(f"Oracle client path does not exist: {oracle_client_path}")
os.environ["PATH"] = oracle_client_path + ";" + os.environ["PATH"]

# Function to fetch data from Oracle
def fetch_data_from_oracle(user, password, host, port, service_name, query):
    dsn = URL.create(
        "oracle+cx_oracle",
        username=user,
        password=password,
        host=host,
        port=port,
        database=service_name,
    )
    engine = create_engine(dsn)
    
    with engine.connect() as connection:
        result = connection.execution_options(stream_results=True).execute(text(query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    engine.dispose()
    return df

# Oracle connection parameters
user = 'Alikhan'  
password = 'I7juGGsz'  
host = '10.10.120.96'  
port = 1521
service_name = 'ORCL'

# Определение SQL-запроса
sql_query = """
WITH NP AS (
    SELECT
        rec.OBJECTCODE,
        item.prodnum,
        item.tanknum,
        TO_CHAR(rec.STARTTIMESTAMP, 'dd.mm.yyyy') AS da,
        TO_CHAR(rec.STARTTIMESTAMP, 'hh24') AS hour,
        SUM(item.Volume) AS volume
    FROM
        BI.RECEIPTS rec
    RIGHT JOIN BI.ZREPORTS z 
        ON (z.OBJECTCODE = rec.objectcode AND 
            z.ZREPORTNUM = rec.znum AND 
            z.STARTTIMESTAMP = rec.zSTARTTIMESTAMP)
    LEFT JOIN BI.RECEIPTITEMS item 
        ON item.RECEIPTID = rec.id -- получение продаж в разбивке по товарам
    WHERE 
        EXTRACT(YEAR FROM z.STARTTIMESTAMP) = 2024 
        AND EXTRACT(month FROM z.STARTTIMESTAMP) IN (6,7,8,9)
        AND rec.objectcode = 'F111' 
        AND ITEM.PRODTYPE = 0 
        AND (
            item.BRUTO <> 0 
            OR (
                item.VOLUME <> 0 
                AND ITEM.PRODTYPE = 0 
                AND rec.objectcode IN ('X347', 'X345', 'X343', 'X364', 'X344', 'X348', 'X342', 'X351', 'X349', 'X341', 'X346', 'X350')
            )
        )
    GROUP BY 
        rec.OBJECTCODE,
        item.prodnum, 
        item.tanknum, 
        TO_CHAR(rec.STARTTIMESTAMP, 'dd.mm.yyyy'),
        TO_CHAR(rec.STARTTIMESTAMP, 'hh24')
),

MaxMinutes AS (
    SELECT 
        t.objectcode,
        t.gasnum, 
        t.tank,
        TO_CHAR(t.postimestamp, 'dd.mm.yyyy') AS day,
        TO_CHAR(t.postimestamp, 'hh24') AS hour,
        MAX(TO_CHAR(t.postimestamp, 'mi')) AS max_minute,
        MIN(TO_CHAR(t.postimestamp, 'mi')) AS min_minute
    FROM 
        BI.tigmeasurements t
    WHERE 
        t.objectcode = 'F111' 
        AND t.postimestamp BETWEEN TO_DATE('2024-06-01', 'YYYY-MM-DD') AND TO_DATE('2024-09-17', 'YYYY-MM-DD')
    GROUP BY 
        t.objectcode,
        t.gasnum, 
        t.tank,
        TO_CHAR(t.postimestamp, 'dd.mm.yyyy'),
        TO_CHAR(t.postimestamp, 'hh24')
),

cte AS (
    SELECT 
        t.objectcode,
        t.gasnum, 
        t.tank,
        t.POSTIMESTAMP,
        LEAD(TO_CHAR(t.POSTIMESTAMP, 'dd.mm.yyyy hh24:mi'), 1) OVER (PARTITION BY t.objectcode, t.gasnum, t.tank ORDER BY t.POSTIMESTAMP) AS tr1,
        TO_CHAR(t.POSTIMESTAMP, 'dd.mm.yyyy hh24:mi') AS tr0,
        LAG(TO_CHAR(t.POSTIMESTAMP, 'dd.mm.yyyy hh24:mi'), 1) OVER (PARTITION BY t.objectcode, t.gasnum, t.tank ORDER BY t.POSTIMESTAMP) AS tr2,
        LEAD(t.volume, 1) OVER (PARTITION BY t.objectcode, t.gasnum, t.tank ORDER BY t.POSTIMESTAMP) AS vm1,
        t.volume AS vm0, 
        LAG(t.volume, 1) OVER (PARTITION BY t.objectcode, t.gasnum, t.tank ORDER BY t.POSTIMESTAMP) AS vm2,
        LEAD(TO_CHAR(t.POSTIMESTAMP, 'mi'), 1) OVER (PARTITION BY TO_CHAR(t.POSTIMESTAMP, 'dd.mm.yyyy hh24:mi'), t.objectcode, t.gasnum, t.tank ORDER BY t.POSTIMESTAMP) AS tr11,
        TO_CHAR(t.POSTIMESTAMP, 'mi') AS tr01,
        LAG(TO_CHAR(t.POSTIMESTAMP, 'mi'), 1) OVER (PARTITION BY TO_CHAR(t.POSTIMESTAMP, 'dd.mm.yyyy hh24:mi'), t.objectcode, t.gasnum, t.tank ORDER BY t.POSTIMESTAMP) AS tr21
    FROM BI.tigmeasurements t
    WHERE 
        EXTRACT(YEAR FROM t.POSTIMESTAMP) = 2024 
        AND EXTRACT(month FROM t.POSTIMESTAMP) IN (6,7,8,9)
        AND t.OBJECTCODE = 'F111'
        AND t.GASNUM = 3300000010
        AND t.TANK = 4
),

cte1 AS (
    SELECT 
        t.objectcode,
        t.gasnum, 
        t.tank,
        TO_CHAR(t.POSTIMESTAMP, 'dd.mm.yyyy') AS pstm, 
        TO_CHAR(t.POSTIMESTAMP, 'hh24') AS hh,
        TO_CHAR(t.POSTIMESTAMP, 'mi') AS mi,
        t.tr1, t.tr0, t.tr2, t.vm1, t.vm0, t.vm2,
        CASE WHEN (60 - t.tr21) > t.tr01 THEN t.vm0 ELSE t.vm2 END AS min_value,
        CASE WHEN (60 - t.tr01) > t.tr11 THEN t.vm1 ELSE t.vm0 END AS max_value
    FROM cte t
),

mx AS (
    SELECT 
        t.objectcode,
        t.gasnum, 
        t.tank,
        t.day,
        t.hour,
        c.max_value
    FROM cte1 c
    JOIN MaxMinutes t
        ON c.mi = t.max_minute
        AND c.hh = t.hour
        AND c.pstm = t.day
        AND t.objectcode = c.objectcode
        AND t.gasnum = c.gasnum
        AND t.tank = c.tank
),

mn AS (
    SELECT 
        t.objectcode,
        t.gasnum, 
        t.tank,
        t.day,
        t.hour, 
        c.min_value
    FROM cte1 c
    JOIN MaxMinutes t
        ON c.mi = t.min_minute
        AND c.hh = t.hour
        AND c.pstm = t.day
        AND t.objectcode = c.objectcode
        AND t.gasnum = c.gasnum
        AND t.tank = c.tank
)

SELECT 
    m1.objectcode,
    m1.gasnum,
    m1.tank,
    m1.day,
    m1.hour,
    CASE WHEN (m2.min_value - m1.max_value) < 0 THEN n.volume ELSE (m2.min_value - m1.max_value) END AS diff
FROM mx m1 
JOIN mn m2
    ON m1.day = m2.day
    AND m1.hour = m2.hour
    AND m1.objectcode = m2.objectcode
    AND m1.gasnum = m2.gasnum
    AND m1.tank = m2.tank
JOIN NP n
    ON n.objectcode = m1.objectcode
    AND n.prodnum = m1.gasnum
    AND n.tanknum = m1.tank
    AND n.da = m1.day
    AND n.hour = m1.hour
"""

# Извлечение данных
df = fetch_data_from_oracle(user, password, host, port, service_name, sql_query)

# Переименование колонок (если необходимо)
df.columns = ['АЗС_CODE', 'PRODNAME', 'TANKNUM', 'ДАТА', 'HOUR', 'КОЛИЧЕСТВО']

# Запись в CSV
df.to_csv('F111.csv', index=False, encoding='utf-8')

print("Данные успешно записаны в output.csv")
