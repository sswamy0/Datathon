# Claude Code Efficiency Protocol (Token-Saving Mode)

As an AI assistant operating on a Claude Pro plan, you must prioritize **token efficiency** and **context hygiene** to maximize the 5-hour message quota. Follow these operational constraints:

## 1. Context Management (CRITICAL)
- **Compact Often:** Automatically suggest running `/compact` if the conversation exceeds 15-20 turns.
- **Clearance:** After successfully completing a discrete task (e.g., a bug fix or a specific function implementation), remind the user to run `/clear` before starting the next task.
- **Minimal File Reading:** Do not `cat` entire files unless necessary. Use `grep` or `sed` to read specific lines/functions to keep the context window small.

## 2. Planning over Execution
- **Plan First:** For any task involving changes to >2 files, enter **Plan Mode** first. 
- **Verification:** Outline the logic of the change and wait for user confirmation before executing `write_to_file`. 
- **No Refactoring Loops:** Do not suggest broad refactors unless they are the primary goal of the task.
- **State Synchronization:** Every time a model is trained or a new insight is found, automatically update SUMMARY.md with the metric (e.g., RMSE, F1-score) and the filename of the generating script.

## 3. Tooling & Optimization
- **Specific Sub-agents:** When using Everything-Claude-Code sub-agents, call them for a single specialized purpose (e.g., `@tdd-expert` for a test suite) and then immediately return to the main agent.
- **MCP Hygiene:** Only utilize MCP tools that are strictly relevant to the current prompt.

## 4. Academic/Technical Specifics
- **Optimization Focus:** When writing optimization models, prioritize Python (Pyomo) or R as established in the project environment.
- **Math Accuracy:** Ensure all LaTeX formulas for combinatorial optimization or graph theory are strictly accurate. If unsure of a specific property (e.g., convex partition logic), ask for clarification rather than iterating.
-**Zero-Hallucination Policy:** If a library version or data column name is unknown, use list(df.columns) or pip show rather than guessing. Do not omit existing logic when refactoring; strictly preserve user-defined constraints.

## 5. Build & Test Commands
- **Build:** [Insert your build command, e.g., `make` or `npm run build`]
- **Test:** [Insert your test command, e.g., `pytest` or `npm test`]

## Data Handling Constraints
- **Never `cat` or `read` full datasets:** Always use `df.head()`, `df.info()`, or `df.describe()` to understand schema.
- **Sampling:** When prototyping models, always suggest a 5-10% sample of the data first to verify the pipeline before running on the full set.
- **Path Awareness:** Data is located in `./data/`. Never move or rename these files unless directed.

## Visualization Protocol
- **Headless Mode:** Always use `matplotlib.use('Agg')` for Python scripts.
- **File Output:** Save all plots/charts to the `./outputs/plots/` directory as .png files.
- **Summary Statistics:** After generating a plot, provide a 2-line text summary of the key trend or outlier identified.


## Code Structure
- **Modularization:** Break the pipeline into discrete scripts: `01_eda.py`, `02_preprocessing.py`, `03_model_training.py`, and `04_inference.py`.
- **Intermediary Saves:** Save processed dataframes as `.parquet` or `.csv` between stages so each script can be run (and debugged) in isolation.
- **State Management:** Maintain a `SUMMARY.md` file (you update this) tracking current best model performance (accuracy, RMSE, etc.) so we don't lose track of progress.

## Technical Stack & Reproducibility
- **Libraries:** Prioritize Python (Pandas, Scikit-Learn, Pyomo for optimization tasks, XGBoost/LightGBM).
- **Seeding:** Always set `random_state=42` (or similar) for all stochastic processes to ensure reproducibility.
- **Error Handling:** When a script fails, use `traceback` to identify the specific line and offer a fix immediately.