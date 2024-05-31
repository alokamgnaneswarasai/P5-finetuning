all_tasks = {}

# =============================================================================
# Task: all_bookreads_templates for rating prediction
# =============================================================================

task_subgroup_1 = {}
template = {}

template['source'] = "which start rating will user_{} give to item_{}?"
template['target'] = "{}"
template['task'] ="rating"
template['source_argc'] = 2
template['target_argc'] = 1
template['source_argv'] = ['user_id', 'item_id']
template['target_argv'] = ['star_rating']
template['id'] = "1-1"
task_subgroup_1["1-1"] = template

all_tasks['rating'] = task_subgroup_1

# =============================================================================
# Task: all_bookreads_templates for rating prediction
# =============================================================================

