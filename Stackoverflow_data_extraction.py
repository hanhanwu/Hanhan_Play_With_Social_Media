# extract questions and answers from StackOverflow
import stackexchange as se
from selenium import webdriver
import time
import json
import collections


def get_question_lists(output_path):
    so = se.Site(se.StackOverflow, "[your api key]")

    pagesize = 100
    tagged = "python"
    fromdate = 1488326400  # 2017/3/1
    todate = 1489536000    # 2017/3/15

    question_dct = {}


    for q in so.questions(pagesize=pagesize, tagged=tagged, page=1, fromdate = fromdate, todate = todate,
                          min = fromdate, max = todate):
        question_dct[q.id] = q.link

    with open(output_path, 'a') as out:
        for qid, qlink in question_dct.items():
            out.write(str(qid) + ", " + qlink + "\n")


# get question, answers and answer comments data
def get_post_data(qurl, output_file):
    driver = webdriver.Firefox()
    driver.get(qurl)
    time.sleep(5)
    try:
        # get question data
        q_title = driver.find_element_by_class_name("question-hyperlink").text

        mainbar = driver.find_element_by_id("mainbar")

        q_body = mainbar.find_element_by_class_name("question")
        q_id = q_body.get_attribute("data-questionid")
        q_vote_count = int(q_body.find_element_by_class_name("vote-count-post ").text)
        q_favorite_count = int(q_body.find_element_by_class_name("favoritecount").text)

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


        # get answers data
        answers_body = mainbar.find_element_by_id("answers")
        answers = answers_body.find_elements_by_class_name("answer")

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

        with open(output_file, 'a') as out:
            json.dump(collections.OrderedDict(post_dct), out)

    except:
        print(qurl)
        driver.quit()


def main():
    questions_urls = [
        "http://stackoverflow.com/questions/4843173/how-to-check-if-type-of-a-variable-is-string/35926059#35926059"
    ]
    output = "output/post_4843173.json"

    for qurl in questions_urls:
        get_post_data(qurl, output)

if __name__ == "__main__":
    main()
