#!/usr/bin/env bash

xsb --quietload --noprompt --nofeedback --nobanner << END_XSB_STDIN
set_prolog_flag(unknown, fail).
['$RULES_DIR/general_rules'].
['$VIEWS_DIR/yw_views'].
[user].
:- table q/1.
q(OutputName) :-
    yw_workflow_script(WorkflowId,_,_,_),
    yw_step_output(WorkflowId, _, _, _, _,_, OutputName).
end_of_file.
printall(q(_)).
END_XSB_STDIN
