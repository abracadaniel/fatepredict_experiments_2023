"""Reads the outputs of a watershed + waterz segmentation + agglomeration, and
writes them to a MongoDB database.

We recommend putting the database credentials in a YAML file like below:

```
name: "fatepredict_alice"
user: "username"
pass: "password"
host: "mongodb-server"
port: 12345
```
"""
import pymongo


def create_cells_client(name, host, port, username, password):
    # Creates client
    client = pymongo.MongoClient(host=host)  # TODO add username and password
    db = client[name]
    create_indices = 'nodes' not in db.list_collection_names()
    cells = db['nodes']

    if create_indices:
        # Create indices for quick lookup
        cells.create_index(
            [
                (loc, pymongo.ASCENDING)
                for loc in ['t', 'z', 'y', 'x']
            ],
            name='position')
        cells.create_index(
            [
                ('id', pymongo.ASCENDING)
            ],
            name='id',
            unique=True)
    return client, cells


if __name__ == "__main__":
    pass
