-- Statements share one session; this comment has a ; in it.
CREATE TABLE foods AS
SELECT category, calories FROM read_csv('../../examples/datasets/foods1.csv');

SELECT category, SUM(calories) AS calories
FROM foods
GROUP BY category
ORDER BY category;

/* The last statement needs no semicolon. */
SELECT COUNT(*) AS n FROM foods WHERE category = 'fruit'
