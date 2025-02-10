# Generating Polars code with LLMs
Large Language Models (LLMs) can sometimes return Pandas code or invalid Polars code in their output. Here we sets out some approaches that help LLMs to generate valid Polars code more consistently. 

These approaches have been developed by the Polars community by evaluating model responses to a variety of inputs. Raise an [issue](https://github.com/pola-rs/polars/issues) if you find an alternative approache for generating Polars code from LLMs. 


## System prompt
For many LLMs you can provide a system prompt that is added to every individual prompt you send to a model. In the system prompt you can specify your preferred defaults. For example you can include "Use Polars as the default dataframe library". Including a system prompt along these lines typically leads to models consistently generating code in Polars rather than Pandas.

This system prompt can be set as a customization option within the settings menu for both web-based LLMs such as Chatgpt.com or for IDE-based LLMs such as Cursor. See the documentation for each app to confirm the current interface.

## Enable web search
Some LLMs can search the web to find information beyond what is in the prompt and in their pre-training data. Enabling web search on a query can allow an LLM to search the Polars documentation to reference the up-to-date API.

Unfortunately, enabling web search is not a universal solution. From our experience it appears that if a model is confident in a result based on its pre-training data than it does not incorporate a web search in its output. 

## Provide examples
You can guide LLMs to the correct syntax by providing relevant examples of the functionality or syntax you want in your prompt. 

For example, if we provide the following query:
```python
df = pl.DataFrame({
    "id": ["a", "b", "a", "b", "c"],
    "score": [1, 2, 1, 3, 3],
    "year": [2020, 2020, 2021, 2021, 2021],
})
# Compute average of score by id
```
We generally get a result with out-of-date `groupby` syntax instead of `group_by`.

However, if we do the following query (preferably with web search enabled) with an example taken from the Polars `group_by` docstrings:
```python
df = pl.DataFrame({
    "id": ["a", "b", "a", "b", "c"],
    "score": [1, 2, 1, 3, 3],
    "year": [2020, 2020, 2021, 2021, 2021],
})
# Compute average of score by id
# Examples of Polars code:

# df.group_by("a").agg(pl.col("b").mean())
```
We get valid outputs much more often. This basic idea has been validated against a number of leading models.

From the output of models that describe their reasoning process it appears that the combination of web search and examples is particularly effective. If the example contradicts the output expected by the model from its pre-training it seems to trigger the model to do a web search to confirm the results.

Providing explicit instructions such as "use `group_by` instead of `groupby`" can also be effective.


