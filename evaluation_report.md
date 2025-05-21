# NBA Stats NL-to-SQL Evaluation Report
Generated on: 2025-05-09 13:13:50

## Executive Summary
- Total queries evaluated: 98
- Exact match rate improved by 6.12% (from 0.00% to 6.12%)
- High similarity rate improved by 34.69% (from 0.00% to 34.69%)
- Most improved query type: 'detail' with 14.29% increase in exact matches

## Overall Performance
| Metric               | Baseline   | Improved   | Difference   | % Change   |
|:---------------------|:-----------|:-----------|:-------------|:-----------|
| Exact Match Rate     | 0.00%      | 6.12%      | 6.12%        | 6122.45%   |
| High Similarity Rate | 0.00%      | 34.69%     | 34.69%       | 34693.88%  |

## Performance by Query Type
| Query Type   |   Sample Size | Baseline Exact Match   | Improved Exact Match   | Difference   | % Change   |
|:-------------|--------------:|:-----------------------|:-----------------------|:-------------|:-----------|
| detail       |             7 | 0.00%                  | 14.29%                 | 14.29%       | 14285.71%  |
| counting     |            19 | 0.00%                  | 10.53%                 | 10.53%       | 10526.32%  |
| aggregation  |            25 | 0.00%                  | 8.00%                  | 8.00%        | 8000.00%   |
| filtering    |            13 | 0.00%                  | 7.69%                  | 7.69%        | 7692.31%   |
| comparison   |             3 | 0.00%                  | 0.00%                  | 0.00%        | 0.00%      |
| ranking      |            30 | 0.00%                  | 0.00%                  | 0.00%        | 0.00%      |
| history      |             1 | 0.00%                  | 0.00%                  | 0.00%        | 0.00%      |

## Error Analysis
| Error Type          |   Baseline Count |   Improved Count |   Difference | % Change   |
|:--------------------|-----------------:|-----------------:|-------------:|:-----------|
| wrong_table         |               98 |               31 |          -67 | -68.37%    |
| wrong_column        |                0 |               56 |           56 | 5600.00%   |
| missing_join        |                0 |                0 |            0 | 0.00%      |
| incorrect_aggregate |                0 |                2 |            2 | 200.00%    |
| incorrect_condition |                0 |                2 |            2 | 200.00%    |
| other               |                0 |                1 |            1 | 100.00%    |

## Examples of Major Improvements

### Example 1: Which team had the most nationally televised games?
- Query Type: aggregation
- Similarity Score: 0.03 → 0.84 (↑ 0.81)

**Ground Truth SQL:**
```sql
SELECT t.full_name FROM game_summary gs JOIN team t ON gs.home_team_id = t.id OR gs.visitor_team_id = t.id WHERE gs.natl_tv_broadcaster_abbreviation IS NOT NULL GROUP BY t.id, t.full_name ORDER BY COUNT(*) DESC LIMIT 1
```

**Baseline Generated SQL:**
```sql
SELECT team_name, COUNT(*) AS nationally_televised_games
FROM games
WHERE nationally_televised = TRUE
GROUP BY team_name
ORDER BY nationally_televised_games DESC
LIMIT 1;
```

**Improved Generated SQL:**
```sql
SELECT t.full_name, COUNT(*) as national_tv_games FROM game_summary gs JOIN team t ON gs.home_team_id = t.id OR gs.visitor_team_id = t.id WHERE gs.natl_tv_broadcaster_abbreviation IS NOT NULL AND gs.natl_tv_broadcaster_abbreviation != '' GROUP BY t.id, t.full_name ORDER BY national_tv_games DESC LIMIT 1
```

### Example 2: Which team has the highest field goal percentage in home games?
- Query Type: ranking
- Similarity Score: 0.03 → 0.83 (↑ 0.79)

**Ground Truth SQL:**
```sql
SELECT t.full_name FROM game g JOIN team t ON g.team_id_home = t.id GROUP BY t.id, t.full_name HAVING COUNT(*) >= 20 ORDER BY CAST(SUM(g.fgm_home) AS FLOAT) / SUM(g.fga_home) * 100 DESC LIMIT 1
```

**Baseline Generated SQL:**
```sql
SELECT 
    team_name,
    SUM(field_goals_made) * 100.0 / NULLIF(SUM(field_goals_attempted), 0) AS field_goal_percentage
FROM 
    games
WHERE 
    is_home_game = TRUE
GROUP BY 
    team_name
ORDER BY 
    field_goal_percentage DESC
LIMIT 1;
```

**Improved Generated SQL:**
```sql
SELECT t.full_name, ROUND(AVG(g.fg_pct_home) * 100, 2) AS avg_fg_pct_home FROM game g JOIN team t ON g.team_id_home = t.id GROUP BY t.id, t.full_name ORDER BY avg_fg_pct_home DESC LIMIT 1
```

### Example 3: What's the most common jersey number?
- Query Type: ranking
- Similarity Score: 0.10 → 0.89 (↑ 0.78)

**Ground Truth SQL:**
```sql
SELECT jersey FROM common_player_info WHERE jersey != '' GROUP BY jersey ORDER BY COUNT(*) DESC LIMIT 1
```

**Baseline Generated SQL:**
```sql
SELECT jersey_number, COUNT(*) AS frequency
FROM players
GROUP BY jersey_number
ORDER BY frequency DESC
LIMIT 1;
```

**Improved Generated SQL:**
```sql
SELECT jersey, COUNT(*) as count FROM common_player_info WHERE jersey != '' GROUP BY jersey ORDER BY count DESC LIMIT 1
```

## Examples of Still Problematic Queries

### Example 1: Which school has the best average draft position?
- Query Type: aggregation
- Similarity Score: 0.00 → 0.06

**Ground Truth SQL:**
```sql
SELECT dh.organization FROM draft_history dh WHERE dh.organization_type = 'College/University' GROUP BY dh.organization HAVING COUNT(*) >= 5 ORDER BY AVG(dh.overall_pick) ASC LIMIT 1
```

**Baseline Generated SQL:**
```sql
SELECT 
    school,
    AVG(draft_position) AS average_draft_position
FROM 
    players
WHERE 
    draft_position IS NOT NULL
GROUP BY 
    school
ORDER BY 
    average_draft_position ASC
LIMIT 1;
```

**Improved Generated SQL:**
```sql
SELECT cpi.school, ROUND(AVG(dh.overall_pick), 2) AS avg_draft_position, COUNT(*) AS players_drafted FROM draft_history dh JOIN common_player_info cpi ON dh.person_id = cpi.person_id WHERE cpi.school IS NOT NULL AND cpi.school != '' GROUP BY cpi.school HAVING COUNT(*) >= 5 ORDER BY avg_draft_position ASC LIMIT 10
```

### Example 2: Which team has the best record in close games (margin ≤ 5 points)?
- Query Type: ranking
- Similarity Score: 0.04 → 0.11

**Ground Truth SQL:**
```sql
SELECT t.full_name FROM game g JOIN team t ON t.id IN (g.team_id_home, g.team_id_away) WHERE ABS(g.pts_home - g.pts_away) <= 5 GROUP BY t.id, t.full_name HAVING COUNT(*) >= 100 ORDER BY CAST(SUM(CASE WHEN (g.team_id_home = t.id AND g.pts_home > g.pts_away) OR (g.team_id_away = t.id AND g.pts_away > g.pts_home) THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 DESC LIMIT 1
```

**Baseline Generated SQL:**
```sql
WITH close_games AS (
    SELECT 
        home_team AS team,
        CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS win
    FROM games
    WHERE ABS(home_score - away_score) <= 5
    
    UNION ALL
    
    SELECT 
        away_team AS team,
        CASE WHEN away_score > home_score THEN 1 ELSE 0 END AS win
    FROM games
    WHERE ABS(home_score - away_score) <= 5
),
team_records AS (
    SELECT 
        team,
        SUM(win) AS wins,
        COUNT(*) - SUM(win) AS losses,
        SUM(win) * 1.0 / COUNT(*) AS win_percentage
    FROM close_games
    GROUP BY team
)
SELECT 
    team,
    wins,
    losses,
    win_percentage
FROM team_records
WHERE wins > 0  -- Only include teams that have at least one win
ORDER BY win_percentage DESC, wins DESC
LIMIT 1;
```

**Improved Generated SQL:**
```sql
WITH close_games AS (
  SELECT 
    team_id_home AS team_id, 
    CASE WHEN wl_home = 'W' THEN 1 ELSE 0 END AS win
  FROM game 
  WHERE ABS(pts_home - pts_away) <= 5
  UNION ALL
  SELECT 
    team_id_away AS team_id, 
    CASE WHEN wl_away = 'W' THEN 1 ELSE 0 END AS win
  FROM game 
  WHERE ABS(pts_home - pts_away) <= 5
)
SELECT 
  t.full_name, 
  COUNT(*) AS total_close_games,
  SUM(cg.win) AS wins,
  ROUND(CAST(SUM(cg.win) AS FLOAT) / COUNT(*) * 100, 2) AS win_percentage
FROM close_games cg
JOIN team t ON cg.team_id = t.id
GROUP BY cg.team_id, t.full_name
HAVING COUNT(*) >= 10
ORDER BY win_percentage DESC
LIMIT 1
```

### Example 3: What's the win percentage difference between teams with 3+ rest days vs fewer rest days?
- Query Type: comparison
- Similarity Score: 0.06 → 0.20

**Ground Truth SQL:**
```sql
WITH rest_days AS (SELECT g.game_id, t.id, t.full_name, JULIANDAY(g.game_date) - JULIANDAY(LAG(g.game_date) OVER (PARTITION BY t.id ORDER BY g.game_date)) as days_rest, CASE WHEN (g.team_id_home = t.id AND g.pts_home > g.pts_away) OR (g.team_id_away = t.id AND g.pts_away > g.pts_home) THEN 1 ELSE 0 END as won FROM game g JOIN team t ON t.id IN (g.team_id_home, g.team_id_away)) SELECT ROUND(AVG(CASE WHEN days_rest >= 3 THEN won END) * 100, 2) - ROUND(AVG(CASE WHEN days_rest < 3 THEN won END) * 100, 2) as win_pct_diff FROM rest_days WHERE days_rest IS NOT NULL
```

**Baseline Generated SQL:**
```sql
WITH RestDayStats AS (
    SELECT 
        CASE WHEN rest_days >= 3 THEN 'ThreeOrMoreRestDays' ELSE 'FewerRestDays' END AS rest_category,
        COUNT(*) AS total_games,
        SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins
    FROM games
    GROUP BY CASE WHEN rest_days >= 3 THEN 'ThreeOrMoreRestDays' ELSE 'FewerRestDays' END
)

SELECT 
    MAX(CASE WHEN rest_category = 'ThreeOrMoreRestDays' THEN (wins * 100.0 / total_games) END) AS three_plus_rest_win_pct,
    MAX(CASE WHEN rest_category = 'FewerRestDays' THEN (wins * 100.0 / total_games) END) AS fewer_rest_win_pct,
    MAX(CASE WHEN rest_category = 'ThreeOrMoreRestDays' THEN (wins * 100.0 / total_games) END) - 
    MAX(CASE WHEN rest_category = 'FewerRestDays' THEN (wins * 100.0 / total_games) END) AS win_pct_difference
FROM RestDayStats;
```

**Improved Generated SQL:**
```sql
WITH game_with_rest AS (
  SELECT 
    g1.game_id,
    g1.team_id_home,
    g1.team_id_away,
    g1.wl_home,
    g1.wl_away,
    JULIANDAY(g1.game_date) - JULIANDAY(MAX(g2.game_date)) AS home_rest_days,
    JULIANDAY(g1.game_date) - JULIANDAY(MAX(g3.game_date)) AS away_rest_days
  FROM game g1
  LEFT JOIN game g2 ON (g1.team_id_home = g2.team_id_home OR g1.team_id_home = g2.team_id_away)
    AND g2.game_date < g1.game_date
  LEFT JOIN game g3 ON (g1.team_id_away = g3.team_id_home OR g1.team_id_away = g3.team_id_away)
    AND g3.game_date < g1.game_date
  GROUP BY g1.game_id, g1.team_id_home, g1.team_id_away
),
win_stats AS (
  SELECT
    ROUND(SUM(CASE WHEN home_rest_days >= 3 AND wl_home = 'W' THEN 1 ELSE 0 END) * 100.0 / 
      NULLIF(SUM(CASE WHEN home_rest_days >= 3 THEN 1 ELSE 0 END), 0), 2) AS home_win_pct_more_rest,
    ROUND(SUM(CASE WHEN home_rest_days < 3 AND wl_home = 'W' THEN 1 ELSE 0 END) * 100.0 / 
      NULLIF(SUM(CASE WHEN home_rest_days < 3 THEN 1 ELSE 0 END), 0), 2) AS home_win_pct_less_rest,
    ROUND(SUM(CASE WHEN away_rest_days >= 3 AND wl_away = 'W' THEN 1 ELSE 0 END) * 100.0 / 
      NULLIF(SUM(CASE WHEN away_rest_days >= 3 THEN 1 ELSE 0 END), 0), 2) AS away_win_pct_more_rest,
    ROUND(SUM(CASE WHEN away_rest_days < 3 AND wl_away = 'W' THEN 1 ELSE 0 END) * 100.0 / 
      NULLIF(SUM(CASE WHEN away_rest_days < 3 THEN 1 ELSE 0 END), 0), 2) AS away_win_pct_less_rest
  FROM game_with_rest
)
SELECT 
  ROUND(((home_win_pct_more_rest + away_win_pct_more_rest) / 2) - 
        ((home_win_pct_less_rest + away_win_pct_less_rest) / 2), 2) AS win_pct_difference
FROM win_stats
```

## Examples of Regressions

### Example 1: How many players are from Kentucky?
- Query Type: counting
- Similarity Score: 0.64 → 0.37 (↓ 0.28)

**Ground Truth SQL:**
```sql
SELECT COUNT(*) FROM draft_history WHERE organization = 'Kentucky'
```

**Improved Generated SQL:**
```sql
SELECT COUNT(*) as kentucky_players FROM common_player_info WHERE school = 'Kentucky' LIMIT 1
```

**Baseline Generated SQL:**
```sql
SELECT COUNT(*) 
FROM players 
WHERE state = 'Kentucky';
```

## Conclusions and Recommendations
Based on the evaluation, we can draw the following conclusions:
1. The improved prompt significantly enhanced performance, with a 6.12% increase in exact matches.
2. The query types that saw the most improvement were:
   - detail: +14.29%
   - counting: +10.53%
   - aggregation: +8.00%
3. Overall errors reduced by 6 (6.12%)

### Recommendations for Further Improvement
1. Focus on improving handling of these query types:
   - ranking (7 problematic examples)
   - aggregation (4 problematic examples)
   - filtering (3 problematic examples)
2. Consider adding few-shot examples for complex query types
3. Further enhance the prompt with specific guidance on database schema relationships
4. Test with a more recent Claude model for potentially better performance