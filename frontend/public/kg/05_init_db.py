import sqlite3

conn = sqlite3.connect('05_graph-r2rml.db')
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS works_in;
DROP TABLE IF EXISTS person;
DROP TABLE IF EXISTS department;

CREATE TABLE person (
  id INTEGER PRIMARY KEY,
  name TEXT,
  birth_date TEXT
);

CREATE TABLE department (
  dept_id INTEGER PRIMARY KEY,
  dept_name TEXT
);

CREATE TABLE works_in (
  person_id INTEGER,
  dept_id INTEGER,
  FOREIGN KEY(person_id) REFERENCES person(id),
  FOREIGN KEY(dept_id) REFERENCES department(dept_id)
);

INSERT INTO person VALUES (1, 'Alice', '1990-01-01');
INSERT INTO person VALUES (2, 'Bob', '1985-05-12');
INSERT INTO department VALUES (10, 'HR');
INSERT INTO department VALUES (20, 'Engineering');
INSERT INTO works_in VALUES (1, 10);
INSERT INTO works_in VALUES (2, 20);
""")

conn.commit()
conn.close()
print("Database 05_graph-r2rml.db initialized.")