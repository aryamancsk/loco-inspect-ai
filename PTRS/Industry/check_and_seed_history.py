import sqlite3
from datetime import datetime, timedelta

db_path = "PTRS/Industry/webapp/inspector.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Get demo user id
cur.execute("SELECT id FROM users WHERE username='demo' OR email='demo@nus.edu.sg'")
row = cur.fetchone()
demo_user_id = row[0] if row else 1

now = datetime.now()
samples = [
    (demo_user_id, "juice_bottle", "conservative", "structural_236.png", "ANOMALOUS", "HIGH SEVERITY", 42.851, 36.462, 4.21, 6.8, "Label Zone (Y: 62%, X: 48%)", "Physical puncture & torn label detected in fruit label sector.", (now - timedelta(minutes=14)).isoformat()),
    (demo_user_id, "juice_bottle", "optimal_f1", "logical_094.png", "ANOMALOUS", "MEDIUM SEVERITY", 31.420, 29.977, 1.84, 4.2, "Top Cap / Fill Neck (Y: 15%, X: 50%)", "Improper liquid fill level & missing top security tamper seal.", (now - timedelta(minutes=10)).isoformat()),
    (demo_user_id, "splicing_connectors", "balanced", "logical_042.png", "ANOMALOUS", "HIGH SEVERITY", 38.914, 32.180, 3.12, 5.5, "Terminal Lever #2 (Y: 40%, X: 65%)", "Incorrect lever closure state: Lever 2 remains in upright unlatched position.", (now - timedelta(minutes=3)).isoformat()),
    (demo_user_id, "breakfast_box", "optimal_f1", "good_007.png", "NORMAL", "NORMAL / NOMINAL", 22.840, 28.450, -0.85, 0.0, "None (Nominal Field)", "Conforming product. Component counts verified: 2 nectarines, 1 tangerine present.", (now - timedelta(minutes=1)).isoformat()),
]

cur.executemany("""
    INSERT INTO inspections 
    (user_id, category, operating_mode, filename, decision, severity, score, threshold, z_score, hotspot_pct, peak_region, explanation, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", samples)
conn.commit()
print("Added diverse entries for demo user.")
conn.close()
