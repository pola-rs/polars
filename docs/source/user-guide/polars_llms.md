# Generating Polars code with LLMs
Large Language Models (LLMs) can sometimes return Pandas code or invalid Polars code in their output. This guide presents approaches that help LLMs generate valid Polars code more consistently.

These approaches have been developed by the Polars community through testing model responses to various inputs. If you find additional effective approaches for generating Polars code from LLMs, please raise an [issue](https://github.com/pola-rs/polars/issues).

## System prompt
Many LLMs allow you to provide a system prompt that is included with every individual prompt you send to the model. In the system prompt, you can specify your preferred defaults, such as "Use Polars as the default dataframe library". Including such a system prompt typically leads to models consistently generating Polars code rather than Pandas code.

You can set this system prompt in the settings menu of both web-based LLMs like ChatGPT and IDE-based LLMs like Cursor. Refer to each application's documentation for specific instructions.

## Enable web search
Some LLMs can search the web to access information beyond their pre-training data. Enabling web search allows an LLM to reference up-to-date Polars documentation for the current API.

However, web search is not a universal solution. Our experience suggests that if a model is confident in a result based on its pre-training data, it may not incorporate web search results in its output.

## Provide examples
You can guide LLMs to use correct syntax by including relevant examples in your prompt.

For instance, this basic query:
```python
df = pl.DataFrame({
    "id": ["a", "b", "a", "b", "c"],
    "score": [1, 2, 1, 3, 3],
    "year": [2020, 2020, 2021, 2021, 2021],
})
# Compute average of score by id
```
Often results in outdated `groupby` syntax instead of the correct `group_by`.

However, including an example from the Polars `group_by` documentation (preferably with web search enabled) like this:
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
Produces valid outputs more consistently. This approach has been validated across several leading models.

Based on models that explain their reasoning, the combination of web search and examples appears particularly effective. When an example contradicts the model's pre-trained expectations, it seems more likely to trigger a web search for verification.

Additionally, explicit instructions like "use `group_by` instead of `groupby`" can be effective in guiding the model to use correct syntax.


