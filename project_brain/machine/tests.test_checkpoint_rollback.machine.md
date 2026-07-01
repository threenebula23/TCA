MODULE: tests.test_checkpoint_rollback

PURPOSE:
Откат чата и workspace по turn_index (как снимок до запроса).

PUBLIC_API:


DEPENDENCIES:
- Agent.checkpoint
- Agent.path_utils
- Agent.versioning
- __future__
- langchain_core.messages
- os
- pathlib
- shutil
- sys
- tempfile
- unittest
- unittest.mock

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:

RISKS:
