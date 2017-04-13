# get code metrics, NLP features
# All the metrics should be for each answer
import os
import json
import re
import subprocess
import tempfile
from radon.raw import analyze
from radon.visitors import ComplexityVisitor
from radon.metrics import h_visit, h_visit_ast, mi_visit
import difflib
import sys
from pycorenlp import StanfordCoreNLP
import ast
import csv


# convert code string to .py file, so that pylint could read it
## its returned file_path is the param input of generate_pylint_metrics()
def code2file(code_input, qid, aid, idx):
    file_name = qid + "_" + aid + "_" + str(idx) + ".py"
    file_path = "/Users/devadmin/Documents/886_final_project/output/code_folder/answer_code/" + file_name
    with open(file_path, 'w') as out:
        out.write(code_input)
    return file_path


# code file to pylint metrics file
## its returned output_metrics_path is the input param of read_pylint_metrics()
def generate_pylint_metrics(code_file_path, qid, aid, idx):
    file_name = qid + "_" + aid + "_" + str(idx) + ".txt"
    output_metrics_path = "/Users/devadmin/Documents/886_final_project/output/code_folder/pylint_metrics/" + file_name
    with tempfile.TemporaryFile() as tempf:
        proc = subprocess.Popen(['pylint',
                                 code_file_path,
                                 '--output-format=parseable'], stdout=tempf)
        proc.wait()
        tempf.seek(0)
        output = tempf.read()
        with open(output_metrics_path, 'w') as out:
            out.write(output)
    return output_metrics_path


def make_default_pylint_metrics():
    collected_metrics = {}

    collected_metrics['module_num'] = 0.0
    collected_metrics['module_documented'] = 0.0
    collected_metrics['module_badname'] = 0.0
    collected_metrics['class_num'] = 0.0
    collected_metrics['class_documented'] = 0.0
    collected_metrics['class_badname'] = 0.0
    collected_metrics['function_num'] = 0.0
    collected_metrics['function_documented'] = 0.0
    collected_metrics['function_badname'] = 0.0
    collected_metrics['code_percentage'] = 0.0
    collected_metrics['docstring_percentage'] = 0.0
    collected_metrics['comment_percentage'] = 0.0
    collected_metrics['empty_percentage'] = 0.0
    collected_metrics['duplication_percentage'] = 0.0
    collected_metrics['convention_num'] = 0.0
    collected_metrics['refactor_num'] = 0.0
    collected_metrics['warning_num'] = 0.0
    collected_metrics['error_num'] = 0.0
    collected_metrics['redefine_num'] = 0.0
    collected_metrics['missing_final_newline_num'] = 0.0
    collected_metrics['missing_docstring_newline_num'] = 0.0
    collected_metrics['bad_whitespace_num'] = 0.0
    collected_metrics['pylint_score'] = 0.0

    return collected_metrics


# convert pylint metrics file into metrics dictionary
def read_pylint_metrics(input_file):
    collected_metrics = make_default_pylint_metrics()

    try:
        ptn = "Your code has been rated at (.*?)/10 \(.*?\)"
        with open(input_file) as pylint_metrics:
            for r in pylint_metrics:
                if '|module' in r:
                    elems = r.split('|')
                    collected_metrics['module_num'] = float(elems[2].strip())
                    collected_metrics['module_documented'] = float(elems[5].strip())
                    collected_metrics['module_badname'] = float(elems[6].strip())
                elif '|class' in r:
                    elems = r.split('|')
                    collected_metrics['class_num'] = float(elems[2].strip())
                    collected_metrics['class_documented'] = float(elems[5].strip())
                    collected_metrics['class_badname'] = float(elems[6].strip())
                elif '|function' in r:
                    elems = r.split('|')
                    collected_metrics['function_num'] = float(elems[2].strip())
                    collected_metrics['function_documented'] = float(elems[5].strip())
                    collected_metrics['function_badname'] = float(elems[6].strip())
                elif '|code' in r:
                    collected_metrics['code_percentage'] = float(r.split('|')[3].strip())
                elif '|docstring' in r:
                    collected_metrics['docstring_percentage'] = float(r.split('|')[3].strip())
                elif '|comment' in r:
                    collected_metrics['comment_percentage'] = float(r.split('|')[3].strip())
                elif '|empty' in r:
                    collected_metrics['empty_percentage'] = float(r.split('|')[3].strip())
                elif '|percent duplicated lines' in r:
                    collected_metrics['duplication_percentage'] = float(r.split('|')[2].strip())
                elif '|convention' in r:
                    collected_metrics['convention_num'] = float(r.split('|')[2].strip())
                elif '|refactor' in r:
                    collected_metrics['refactor_num'] = float(r.split('|')[2].strip())
                elif '|warning' in r:
                    collected_metrics['warning_num'] = float(r.split('|')[2].strip())
                elif '|error' in r:
                    collected_metrics['error_num'] = float(r.split('|')[2].strip())
                elif '|redefined-builtin' in r:
                    collected_metrics['redefine_num'] = float(r.split('|')[2].strip())
                elif '|missing-final-newline' in r:
                    collected_metrics['missing_final_newline_num'] = float(r.split('|')[2].strip())
                elif '|missing-docstring' in r:
                    collected_metrics['missing_docstring_newline_num'] = float(r.split('|')[2].strip())
                elif '|bad-whitespace' in r:
                    collected_metrics['bad_whitespace_num'] = float(r.split('|')[2].strip())
                elif 'Your code has been rated at' in r:
                    pylint_score = re.match(ptn, r).group(1)
                    collected_metrics['pylint_score'] = float(pylint_score)
        return collected_metrics
    except:
        return collected_metrics


def make_default_radon_metrics():
    radon_metrics = {"loc":0.0, "lloc":0.0, "sloc":0.0, "comments":0.0, "multi":0.0, "blank":0.0, "single_comments":0.0,
                    "function_num":0.0, "total_function_complexity":0.0, "radon_functions_complexity":0.0,
                     "h1":0.0, "h2":0.0, "N1":0.0, "N2":0.0, "vocabulary":0.0, "length":0.0, "calculated_length":0.0,
                     "volume":0.0, "difficulty":0.0, "effort":0.0, "time":0.0, "bugs":0.0, "Maintainability_Index":0.0
                     }
    return radon_metrics


def generate_radon_metrics(code_input):
    radon_metrics = make_default_radon_metrics()
    try:
        # raw metrics
        raw_metrics = analyze(code_input)
        radon_metrics['loc'] = raw_metrics.loc
        radon_metrics['lloc'] = raw_metrics.lloc
        radon_metrics['sloc'] = raw_metrics.sloc
        radon_metrics['comments'] = raw_metrics.comments
        radon_metrics['multi'] = raw_metrics.multi
        radon_metrics['single_comments'] = raw_metrics.single_comments

        # cyclomatic complexity
        cc = ComplexityVisitor.from_code(code_input)
        radon_metrics['function_num'] = len(cc.functions)
        total_function_complexity = 0.0
        for fun in cc.functions:
            total_function_complexity += fun.complexity
        radon_metrics['total_function_complexity'] = total_function_complexity
        radon_metrics['radon_functions_complexity'] = cc.functions_complexity

        # calculate based on AST tree
        v = h_visit_ast(h_visit(code_input))
        radon_metrics['h1'] = v.h1
        radon_metrics['h2'] = v.h2
        radon_metrics['N1'] = v.N1
        radon_metrics['N2'] = v.N2
        radon_metrics['vocabulary'] = v.vocabulary
        radon_metrics['length'] = v.length
        radon_metrics['calculated_length'] = v.calculated_length
        radon_metrics['volume'] = v.volume
        radon_metrics['difficulty'] = v.difficulty
        radon_metrics['effort'] = v.effort
        radon_metrics['time'] = v.time
        radon_metrics['bugs'] = v.bugs

        # Maintainability Index (MI) based on
        ## the Halstead Volume, the Cyclomatic Complexity, the SLOC number and the number of comment lines
        mi = mi_visit(code_input, multi=True)
        radon_metrics['Maintainability_Index'] = mi

        return radon_metrics
    except:
        return radon_metrics


# compare question code and answer code, generate the score
def match_qa_code(question_code_list, answer_code_list):
    if len(question_code_list) == 0 or len(answer_code_list) == 0:
        return 0.0
    max_score = 0.0
    q_code = "".join(question_code_list)
    for elem in answer_code_list:
        tmp_score = difflib.SequenceMatcher(None, elem, q_code).quick_ratio()
        if tmp_score > max_score:
            max_score = tmp_score
    return max_score


def make_default_sentiment():
    dft_sentiment = {"score":0.0, "Positive":0, "Negative":0, "Neutral":0,
                     "Verypositive":0, "Verynegative":0}
    return dft_sentiment

# calculate sentiment score, counts for comment
def standford_sentiment_comment(text_str, vote):
    cmt_sentiment = make_default_sentiment()
    nlp = StanfordCoreNLP('http://localhost:9000')
    res = nlp.annotate(text_str,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 20000,
                       })
    try:
        total_value = 0.0
        for s in res["sentences"]:
            total_value += float(s["sentimentValue"])
            cmt_sentiment[s["sentiment"]] += 1
        if vote > 0:
            cmt_sentiment['score'] = total_value*vote
        else:
            cmt_sentiment['score'] = total_value
        return cmt_sentiment
    except:
        return cmt_sentiment

# calculate sentiment score, counts for answer
def standford_sentiment_answer(text_str):
    asw_sentiment = make_default_sentiment()
    nlp = StanfordCoreNLP('http://localhost:9000')
    res = nlp.annotate(text_str,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 20000,
                       })
    try:
        total_value = 0.0
        for s in res["sentences"]:
            total_value += float(s["sentimentValue"])
            asw_sentiment[s["sentiment"]] += 1
        asw_sentiment['score'] = total_value
        return asw_sentiment
    except:
        return asw_sentiment


def set_default_answer_features(qid, aid):
    tmp_json = {"qid":qid, "aid":aid, "IsUnderrated":'N', "total_pylint_metrics":{}, "total_radon_metrics":{},
                "qa_code_match":0.0, "vote_percent":0.0, "vote_gap":0.0,
                "comment_sentiment":{"score":0.0, "Positive":0, "Negative":0, "Neutral":0,"Verypositive":0, "Verynegative":0},
                "answer_sentiment":{"score":0.0, "Positive":0, "Negative":0, "Neutral":0, "Verypositive":0, "Verynegative":0}}
    return tmp_json


# get qid, answer head about those have underrated answers
def get_yes_dct(labeled_csv):
    yes_dct = {}
    url_ptn = "http://stackoverflow.com/questions/(\d+)"
    with open(labeled_csv) as csv_in:
        for r in csv_in:
            r = r
            elems = r.split(',')
            qid = re.match(url_ptn, elems[0]).group(1)
            label = elems[1]
            if label == "Y":
                text_sample = ",".join(elems[2:len(elems)])
                yes_dct[qid] = text_sample.strip()
    return yes_dct


# collect all the comments for each answer
def get_answer_comments(input_file):
    ac_dct = {}
    with open(input_file) as answer_comments:
        for r in answer_comments:
            tmp_dct = ast.literal_eval(r)
            qid = tmp_dct.keys()[0]
            answers = tmp_dct[qid]
            ac_dct[qid] = {}
            for aid, comments in answers.items():
                ac_dct[qid][aid] = comments

    return ac_dct



# collect post-answer pairs and generate features for each answer
def generate_features(input_folder_path, labeled_csv, ac_dct):
    yes_dct = get_yes_dct(labeled_csv)
    post_answer_lst = []

    sampleY_lst = []
    sampleY_match_lst = []

    for file_name in os.listdir(input_folder_path):
        file_path = input_folder_path + "/" + file_name
        with open(file_path) as infile:
            post_data = json.load(infile)

            post_id = list(post_data.keys())[0]
            sample = ""
            if post_id in yes_dct.keys():
                has_yes = 1
                sample = yes_dct[post_id]
                if sample.startswith("\"") and sample.endswith("\""):
                    sample = sample[1:-1]
                if sample.startswith(" "):
                    sample = sample[1:]
                sample = sample.replace('""', '"')
                sampleY_lst.append(sample)
            else:
                has_yes = 0


            # ************ QUESTION DATA ****************
            q_title = post_data[post_id]['q_title']
            q_vote_count = post_data[post_id]['q_vote_count']
            q_favorite_count = post_data[post_id]['q_favorite_count']
            q_text_list = post_data[post_id]['q_text_list']
            q_code_list = post_data[post_id]['q_code_list']
            q_content_text = post_data[post_id]['q_content_text']


            # ************ ANSWER DATA ****************
            answers = post_data[post_id]['answers']
            total_vote = 0
            max_vote = 0
            for v in answers.values():
                answer_vote_count = v['answer_vote_count']
                total_vote += answer_vote_count
                if answer_vote_count > max_vote:
                    max_vote = answer_vote_count

            for answer_id, v in answers.items():
                answer_vote_count = v['answer_vote_count']
                answer_content_text = v['answer_content_text']
                answer_code_list = v['answer_code_list']
                answer_text_list = v['answer_text_list']

                qa_json = set_default_answer_features(post_id, answer_id)

                if has_yes == 1 and sample in answer_content_text:
                    qa_json['IsUnderrated'] = 'Y'
                    has_yes = 0
                    sampleY_match_lst.append(sample)

                # get pylint metrics, sum up each elem in code list, but filter out elem less than 10 characteristics
                if len(answer_code_list) == 0:
                    total_pylint_metrics = make_default_pylint_metrics()
                else:
                    idx = 0
                    dct_list = []
                    for code_input in answer_code_list:
                        if len(code_input) < 25: continue
                        code_file_path = code2file(code_input, post_id, answer_id, idx)
                        pylint_metrics_file_path = generate_pylint_metrics(code_file_path, post_id, answer_id, idx)
                        pylint_metrics = read_pylint_metrics(pylint_metrics_file_path)
                        dct_list.append(pylint_metrics)
                        idx += 1
                    if len(dct_list) >= 2:
                        total_pylint_metrics = reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.iteritems()), dct_list)
                    elif len(dct_list) == 0:
                        total_pylint_metrics = make_default_pylint_metrics()
                    else:
                        total_pylint_metrics = dct_list[0]

                # get radon metrics
                if len(answer_code_list) == 0:
                    total_radon_metrics = make_default_radon_metrics()
                else:
                    idx = 0
                    dct_list = []
                    for code_input in answer_code_list:
                        if len(code_input) < 25: continue
                        radon_metrics = generate_radon_metrics(code_input)
                        dct_list.append(radon_metrics)
                        idx += 1
                    if len(dct_list) >= 2:
                        total_radon_metrics = reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.iteritems()), dct_list)
                    elif len(dct_list) == 0:
                        total_radon_metrics = make_default_radon_metrics()
                    else:
                        total_radon_metrics = dct_list[0]

                if len(answer_text_list) == 0:
                    answer_sentiment = make_default_sentiment()
                else:
                    answer_sentiment = standford_sentiment_answer(" ".join(answer_text_list))

                qa_json['total_pylint_metrics'] = total_pylint_metrics
                qa_json['total_radon_metrics'] = total_radon_metrics
                qa_json['qa_code_match'] = match_qa_code(q_code_list, answer_code_list)
                qa_json['answer_sentiment'] = answer_sentiment
                qa_json['vote_percent'] = answer_vote_count/(total_vote+0.1)
                qa_json['vote_gap'] = max_vote-answer_vote_count


                # ************ COMMENT DATA ****************
                comments = ac_dct[post_id][answer_id]['comments']
                if len(comments) == 0:
                    cmt_sentiment = make_default_sentiment()
                else:
                    cmt_sentimemt_lst = []
                    for comment_id, cmt in comments.items():
                        comment_content_text = cmt['comment_content_text']
                        comment_vote_count = float(cmt['comment_vote_count'])
                        comment_code_list = cmt['comment_code_list']

                        part_sentiment = standford_sentiment_comment(comment_content_text, comment_vote_count)
                        cmt_sentimemt_lst.append(part_sentiment)
                    if len(cmt_sentimemt_lst) == 1:
                        cmt_sentiment = cmt_sentimemt_lst[0]
                    else:
                        cmt_sentiment = reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.iteritems()),
                                               cmt_sentimemt_lst)
                qa_json['comment_sentiment'] = cmt_sentiment

                post_answer_lst.append(qa_json)
                print(qa_json)

    return post_answer_lst




def main():
    input_folder_path = "/Users/devadmin/Documents/886_final_project/output/labeled_files_120"
    labeled_csv = "/Users/devadmin/Documents/886_final_project/labeled_urls/labeled_urls_120.csv"
    answer_comments_file = "/Users/devadmin/Documents/886_final_project/output/120_answer_comments.txt"

    output_csv = "/Users/devadmin/Documents/886_final_project/output/csv4analysis/csv120.csv"

    ac_dct = get_answer_comments(answer_comments_file)
    post_answer_lst = generate_features(input_folder_path, labeled_csv, ac_dct)

    csv_cols = ['qid', 'aid', 'vote_gap', 'cmtVerypositive', 'cmtPositive', 'cmtVerynegative', 'cmtNegative',
                'cmtNeutral', 'cmtscore', 'convention_num', 'refactor_num', 'missing_docstring_newline_num',
                'duplication_percentage', 'error_num', 'bad_whitespace_num', 'module_documented', 'module_badname',
                'module_num', 'docstring_percentage', 'redefine_num', 'warning_num', 'function_documented',
                'class_documented', 'class_badname', 'pylint_score', 'empty_percentage', 'comment_percentage',
                'missing_final_newline_num', 'pylint_function_num', 'code_percentage', 'class_num', 'function_badname',
                'vote_percent', 'h2', 'h1', 'bugs', 'radon_functions_complexity', 'calculated_length', 'blank',
                'loc', 'Maintainability_Index', 'single_comments', 'comments', 'sloc', 'total_function_complexity',
                'vocabulary', 'multi', 'volume', 'difficulty', 'N1', 'N2', 'effort', 'lloc', 'radon_function_num',
                'length', 'time', 'aswVerypositive', 'aswPositive', 'aswVerynegative', 'aswNegative',
                'aswNeutral', 'aswscore', 'qa_code_match', 'IsUnderrated']

    with open(output_csv, 'a') as csv_output:
        writer = csv.DictWriter(csv_output, fieldnames=csv_cols)
        writer.writeheader()

        for dct in post_answer_lst:
            writer.writerow({
                'qid':dct['qid'], 'aid':dct['aid'], 'vote_gap':dct['vote_gap'],
                'cmtVerypositive':dct['comment_sentiment']['Verypositive'], 'cmtPositive':dct['comment_sentiment']['Positive'],
                'cmtVerynegative':dct['comment_sentiment']['Verynegative'], 'cmtNegative':dct['comment_sentiment']['Negative'],
                'cmtNeutral':dct['comment_sentiment']['Neutral'], 'cmtscore':dct['comment_sentiment']['score'],
                'convention_num':dct['total_pylint_metrics']['convention_num'],
                'refactor_num':dct['total_pylint_metrics']['refactor_num'],
                'missing_docstring_newline_num':dct['total_pylint_metrics']['missing_docstring_newline_num'],
                'duplication_percentage':dct['total_pylint_metrics']['duplication_percentage'],
                'error_num':dct['total_pylint_metrics']['error_num'],
                'bad_whitespace_num':dct['total_pylint_metrics']['bad_whitespace_num'],
                'module_documented':dct['total_pylint_metrics']['module_documented'],
                'module_badname':dct['total_pylint_metrics']['module_badname'],
                'module_num':dct['total_pylint_metrics']['module_num'],
                'docstring_percentage':dct['total_pylint_metrics']['docstring_percentage'],
                'redefine_num':dct['total_pylint_metrics']['redefine_num'],
                'warning_num':dct['total_pylint_metrics']['warning_num'],
                'function_documented':dct['total_pylint_metrics']['function_documented'],
                'class_documented':dct['total_pylint_metrics']['class_documented'],
                'class_badname':dct['total_pylint_metrics']['class_badname'],
                'pylint_score':dct['total_pylint_metrics']['pylint_score'],
                'empty_percentage':dct['total_pylint_metrics']['empty_percentage'],
                'comment_percentage':dct['total_pylint_metrics']['comment_percentage'],
                'missing_final_newline_num':dct['total_pylint_metrics']['missing_final_newline_num'],
                'pylint_function_num':dct['total_pylint_metrics']['function_num'],
                'code_percentage':dct['total_pylint_metrics']['code_percentage'],
                'class_num':dct['total_pylint_metrics']['class_num'],
                'function_badname':dct['total_pylint_metrics']['function_badname'],
                'vote_percent':dct['vote_percent'], 'h2':dct['total_radon_metrics']['h2'],
                'h1':dct['total_radon_metrics']['h1'],'bugs':dct['total_radon_metrics']['bugs'],
                'radon_functions_complexity':dct['total_radon_metrics']['radon_functions_complexity'],
                'calculated_length':dct['total_radon_metrics']['calculated_length'],
                'blank':dct['total_radon_metrics']['blank'],
                'loc':dct['total_radon_metrics']['loc'], 'Maintainability_Index':dct['total_radon_metrics']['Maintainability_Index'],
                'single_comments':dct['total_radon_metrics']['single_comments'],
                'comments':dct['total_radon_metrics']['comments'], 'sloc':dct['total_radon_metrics']['sloc'],
                'total_function_complexity':dct['total_radon_metrics']['total_function_complexity'],
                'vocabulary':dct['total_radon_metrics']['vocabulary'], 'multi':dct['total_radon_metrics']['multi'],
                'volume':dct['total_radon_metrics']['volume'], 'difficulty':dct['total_radon_metrics']['difficulty'],
                'N1':dct['total_radon_metrics']['N1'], 'N2':dct['total_radon_metrics']['N2'],
                'effort':dct['total_radon_metrics']['effort'], 'lloc':dct['total_radon_metrics']['lloc'],
                'radon_function_num':dct['total_radon_metrics']['function_num'],
                'length':dct['total_radon_metrics']['length'], 'time':dct['total_radon_metrics']['time'],
                'aswVerypositive':dct['answer_sentiment']['Verypositive'], 'aswPositive':dct['answer_sentiment']['Positive'],
                'aswVerynegative':dct['answer_sentiment']['Verynegative'], 'aswNegative':dct['answer_sentiment']['Negative'],
                'aswNeutral':dct['answer_sentiment']['Neutral'], 'aswscore':dct['answer_sentiment']['score'],
                'qa_code_match':dct['qa_code_match'], 'IsUnderrated':dct['IsUnderrated']
            })


if __name__ == "__main__":
    main()
