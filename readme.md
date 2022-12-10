

## Dataset
CIFAR-10

## Step 1. Prepare the architecture dataset on DARTS search space
```shell
bash scripts/gen_graphs.sh
```

## Step 2. Train the graph2vec model
```shell
bash scripts/data_json.sh
```

## Step 3. Search the best architecture on DARTS search space
```shell
bash scripts/search.sh
```