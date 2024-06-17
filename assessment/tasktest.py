import os
import json

def runtest():
    fail_score = 0
    fail_message = '''It doesn't look like you've
                      completed the assessment
                      yet. Please make sure you have
                      completed the assessment notebook
                      in its entirety, then come back
                      and try again.'''

    pass_score = 100
    pass_message = '''Congratulations, you passed
                      the assessment!  Please go to
                      https://courses.nvidia.com/dashboard/
                      and click this course's name at the 
                      top of the dashboard to view
                      your certificate for the workshop.'''

    if os.path.exists('/dli/assessment_results/PASSED'):
        return {'score': pass_score, 'message': pass_message}
    else:
        return {'score': fail_score, 'message': fail_message}


print(json.dumps(runtest()))
