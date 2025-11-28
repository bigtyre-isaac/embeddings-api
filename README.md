Simple Python based API that uses an ML model to compute embeddings for text, assisting in comparing similarity.

# Examples
## POST /api/embeddings/generate
Computes an embedding for some input text.
### Request
```
{
  "text": "Some input text"
}
```

### Response
```
{
    "text": "This is some fancy text.",
    "embedding": [
        -0.02730877324938774,
        0.14506401121616364,
        0.033422309905290604,
        0.004252374637871981,
        0.005175379104912281,
        -0.014435559511184692,
        0.08713068813085556,
      ....
    ]
}
```


## POST /api/similarity/compare-text
Computes embeddings for some input text and a number of candidate values, computes the cosine similarity of each one and returns a response detailing best match and the similarity of each candidate.
### Request
```
{
    "text": "This is some fancy text.",
    "candidates": [
        "A few extravagant words",
        "A bit of fairly plain text",
        "Some pictures",
        "Text about fancy cheese"
    ]
}
```

### Response
```
{
    "metric": "cosine",
    "model": "all-MiniLM-L6-v2",
    "scores": {
        "A bit of fairly plain text": 0.6943178176879883,
        "Text about fancy cheese": 0.5711625814437866,
        "A few extravagant words": 0.42451807856559753,
        "Some pictures": 0.26821473240852356
    },
    "best_match": {
        "index": 1,
        "score": 0.6943178176879883,
        "text": "A bit of fairly plain text"
    },
    "debug": null
}
```
