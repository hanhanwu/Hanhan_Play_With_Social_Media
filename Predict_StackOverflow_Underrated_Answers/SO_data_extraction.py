# extract questions and answers from StackOverflow
import stackexchange as se
from selenium import webdriver
import time
import json
import collections


def get_question_lists(output_path):
    so = se.Site(se.StackOverflow, "[your SO API key]")

    pagesize = 100
    tagged = "python"
    fromdate = 1357776000  # 2013/1/10
    todate = 1397088000    # 2014/4/10

    question_dct = {}


    for q in so.questions(pagesize=pagesize, tagged=tagged, page=1, fromdate = fromdate, todate = todate,
                          min = fromdate, max = todate):
        question_dct[q.id] = q.link

    print(len(question_dct))
    with open(output_path, 'a') as out:
        for qid, qlink in question_dct.items():
            out.write(str(qid) + ", " + qlink + "\n")



# get question, answers and answer comments data
def get_post_data(qid, qurl, n, error_file):
    driver = webdriver.Firefox(executable_path='/Users/devadmin/Documents/geckodriver')
    driver.get(qurl)
    dsession_id = driver.session_id
    if dsession_id == None: driver.quit()
    time.sleep(5)
    try:
        mainbar = driver.find_element_by_id("mainbar")

        # get answers data
        answers_body = mainbar.find_element_by_id("answers")
        answers = answers_body.find_elements_by_class_name("answer")
        answers_len = len(answers)
        if answers_len < n:
            driver.quit()
            return 0

        # get question data
        q_title = driver.find_element_by_class_name("question-hyperlink").text

        q_body = mainbar.find_element_by_class_name("question")
        q_id = q_body.get_attribute("data-questionid")
        q_vote_count = int(q_body.find_element_by_class_name("vote-count-post ").text)
        q_favorite_count = q_body.find_element_by_class_name("favoritecount").text.strip()
        if q_favorite_count == "":
            q_favorite_count = 0
        else:
            q_favorite_count = int(q_favorite_count)

        q_content_text = ""
        q_text_lst, q_code_lst  = [], []
        try:
            q_content = q_body.find_element_by_class_name("post-text")
            q_content_text = q_content.text
            q_text = q_content.find_elements_by_tag_name("p")
            for p in q_text:
                q_text_lst.append(p.text)
            q_code = q_content.find_elements_by_tag_name("code")
            for c in q_code:
                q_code_lst.append(c.text)
        except:
            pass

        post_dct = {}
        post_dct[q_id] = {"q_title":q_title, "q_vote_count":q_vote_count, "q_favorite_count":q_favorite_count,
                          "q_content_text":q_content_text, "q_text_list":q_text_lst, "q_code_list":q_code_lst,
                          "answers":{}}



        for answer in answers:
            a_id = answer.get_attribute("data-answerid")
            a_vote_count = int(answer.find_element_by_class_name("vote-count-post ").text)
            a_content_text = ""
            a_text_lst, a_code_lst  = [], []
            try:
                a_content = answer.find_element_by_class_name("post-text")
                a_content_text = a_content.text
                a_text = a_content.find_elements_by_tag_name("p")
                for p in a_text:
                    a_text_lst.append(p.text)
                a_code = a_content.find_elements_by_tag_name("code")
                for c in a_code:
                    a_code_lst.append(c.text)
            except:
                pass

            # comments for the answer
            comment_dct = {}
            show_more_click = ""
            while(show_more_click != None):
                try:
                    show_more_click = driver.find_element_by_css_selector('.js-show-link.comments-link ')
                    driver.execute_script("arguments[0].scrollIntoView();", show_more_click)
                    show_more_click.click()
                    time.sleep(3)
                except:
                    break
            comments = answer.find_elements_by_class_name("comment")
            for comment in comments:
                comment_vote = 0
                comment_content_text = ""
                comment_code_lst = []
                comment_id = comment.get_attribute("id").split("comment-")[1]
                try:
                    comment_vote = int(comment.find_element_by_class_name(" comment-score").text)
                    comment_content= comment.find_element_by_class_name("comment-copy")
                    comment_content_text = comment_content.text
                    comment_code = comment_content.find_elements_by_tag_name("code")
                    for cmt_code in comment_code:
                        comment_code_lst.append(cmt_code.text)
                except:
                    pass
                comment_dct[comment_id] = {"comment_vote_count":comment_vote,
                                           "comment_content_text":comment_content_text,
                                           "comment_code_list":comment_code_lst}


            post_dct[q_id]["answers"][a_id]  = {"answer_vote_count":a_vote_count, "answer_content_text":a_content_text,
                                                "answer_text_list":a_text_lst, "answer_code_list":a_code_lst,
                                                "comments":comment_dct}

        driver.quit()

        if answers_len == 3:
            output_file = "/Users/devadmin/Documents/886_final_project/output/size3/" + qid + ".json"
        elif answers_len == 4:
            output_file = "/Users/devadmin/Documents/886_final_project/output/size4/" + qid + ".json"
        else:
            output_file = "/Users/devadmin/Documents/886_final_project/output/size5+/" + qid + ".json"

        with open(output_file, 'a') as out:
            json.dump(collections.OrderedDict(post_dct), out)
        return 1

    except:
        with open(error_file, 'a') as exe_out:
            exe_out.write(qid + ", " + qurl)
        driver.quit()
        return 0



def main():
    question_list_output = "/Users/devadmin/Documents/886_final_project/output/question_list.txt"
    error_file = "/Users/devadmin/Documents/886_final_project/output/exception.txt"
    check_ids = "/Users/devadmin/Documents/886_final_project/output/check_ids.txt"

    get_question_lists(question_list_output)    # extract urls from StackExchange API

    # Parse data from urls and generate JSON files for those SO posts contain 3+ answers
    question_dct = {}

    with open(question_list_output) as indata:
        for r in indata:
            elems = r.split(', ')
            q_id = elems[0]
            q_url = elems[1].replace("\n", "")
            question_dct[q_id] = q_url

    sum = 0
    answer_threshold = 3
    checked_id_lst = set()
    with open(check_ids) as checked_in:
        for r in checked_in:
            checked_id_lst.add(r.replace("\n", ""))
    
    for q_id, q_url in question_dct.items():
        if q_id in checked_id_lst: continue
        v = get_post_data(q_id, q_url, answer_threshold, error_file)
        sum += v
        if v != 0:
            print(sum)
        with open(check_ids, 'a') as check_out:
            check_out.write(q_id + "\n")
        time.sleep(5)

if __name__ == "__main__":
    main()
