import sys
sys.path.append('/home/xpeng/research/projects/medicalAI_torch/src/AI-Mod')
import traceback
from src.classifiers.trolling.trolling_classifier import detect_trolling
from src.classifiers.spam.spam_classifier import detect_spam
from src.classifiers.trolling.profanity_detect import detect_profanity
import copy
import pandas as pd
from src.utils.ai_utils import root_path

cheap_classifiers = [detect_trolling, detect_spam, detect_profanity]
empty_report = {'spam': 0, 'troll': 0}
debug_print_type = "none"


def analyze_text(text):
    if len(text) == 0:
        return copy.deepcopy(empty_report)
    final_results = []
    classifiers = cheap_classifiers
    for method in classifiers:
        try:
            result = method(text)
            if debug_print_type in result:
                print(method.__name__ + ": " + str(result[debug_print_type]))
        except:
            traceback.print_exc()
            result = None
        if not isinstance(result, dict):
            print("There's something wrong with the results from %s: %s" % (method.__name__, str(result)))
        else:
            final_results.append(result)

    report = combine_reports(*final_results)
    # turn those decimals into nice integers
    for k, v in report.items():
        if isinstance(v, float):
            report[k] = int(round(v*100.0))

    return report


def combine_reports(*args):
    # if isinstance(args[0], list):
    #     args = args[0]
    master_dict = copy.deepcopy(empty_report) #{'notes': []}
    totals = empty_report.copy()
    for dic in args:
        for k, v in dic.items():
            if k in master_dict:
                if k == 'notes':
                    for note in v:
                        if note not in master_dict['notes']:
                            master_dict['notes'].append(note)
                else:
                    master_dict[k] += v**10
                    totals[k] += 1

    for k, v in totals.items():
        if k != 'notes' and v > 0:
            master_dict[k] = (master_dict[k] / v)**0.1

    return master_dict


if __name__ == '__main__':
    csv_file = root_path + 'data/australian2020_11_24_12_23_8577496.csv'
    csv_file_out = root_path + 'data/australian2020_11_24_12_23_8577496_moderation.csv'
    social_df = pd.read_csv(csv_file, header=0)
    social_df['SPAM_SCORE'] = None
    social_df['TROLLING_SCORE'] = None

    for index, row in social_df.iterrows():
        if row.LANGUAGE == 'English':
            report = analyze_text(row.CONTENT)
            social_df.at[index, 'SPAM_SCORE'] = report['spam']
            social_df.at[index, 'TROLLING_SCORE'] = report['troll']

    social_df.to_csv(csv_file_out, index=False)




