################################################################################
### Step 1
################################################################################
# 필요한 라이브러리
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity # 텍스트 데이터를 벡터화 표츌시키는 라이브러리 이다
#url matching하기
# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl
domain = "openai.com"
full_url = "https://openai.com/"
#특정 url로 부터 하이퍼링크를 fetching 해서 텍스트를 추출 하는경우가 아니면 step1,2,3,4 필요없음
# 1,2,3 은 로컬 도메인 안에잇는 (같은 website 안의 webpages 들만 연결하는 다른 website로 가지않는) 것들만 필터링해서 추출 해내서 예시에 보이는 openai.com만 크롤링 하겟다는 의도.

"""
Retrieve Hyperlinks: The script starts by collecting all the hyperlinks from a given URL (which is declared in section 1) and ensures they are within the same domain.

Save Texts in DataFrame: After retrieving the hyperlinks, the script proceeds to crawl those web pages within the domain. It saves the text content of each web page in a directory and creates a DataFrame to store the text files.

Tokenize and Chunk Texts: The script tokenizes the text using OpenAI's tiktoken package. It then calculates the number of tokens for each text and visualizes the distribution of tokens. If a text exceeds the maximum token limit, it splits it into chunks to accommodate the limit.

Embed Texts: The script uses OpenAI's text embedding API to embed the text chunks or complete texts. The embeddings are stored in the DataFrame for further analysis.

Create Context and Answer Questions: The script creates a context for a given question by finding the most similar context from the DataFrame based on the question. It uses the embeddings and cosine similarity to identify the most relevant context. Finally, it uses OpenAI's GPT-3 model to answer the question based on the selected context.
"""
# hyperlink is clickable links that leads to another webpage
# HTML 파싱(분석하는 단계 / 전체적 구조 분석을 위해 필요ex)elements, contents, tags, attributes)/ 하이퍼링크(리소스 원천ex)텍스트, 이미지 등등) 추출
# url(href attribute) 을 하이퍼 링크 안에 다 넣음
# 하이퍼링크 파서(얻는거임) 함수
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])


################################################################################
### Step 2
################################################################################
#하이퍼 링크는 html 콘텐츠 안의 내용물이고 url은 해당 웹사이트의 주소
# Function to get the hyperlinks from a URL # 하이퍼링크를 URL로 부터 추출 혹은 get
# 하이프링크 파서함수 이용해서 url의html 콘텐츠를 파싱하고 그 콘텐츠를 기반으로 하이퍼링크 를 얻는다
def get_hyperlinks(url):
    # Try to open the URL and read the HTML(tags and attributes)
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser() # 하이퍼 링크 파서 준비
    parser.feed(html)   # html 콘텐츠에 feed해서

    return parser.hyperlinks # 파싱된 하이퍼링크 리턴 (하이퍼링크 얻기 함수를 이용해서 하이퍼링크 얻기)


################################################################################
### Step 3
################################################################################
# 주어진 URL에서 하이퍼링크를 가져오는 함수 구현하기.
# Function to get the hyperlinks from a URL that are within the same domain

def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                    link.startswith("#")
                    or link.startswith("mailto:")
                    or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))
    #같은 도메인에 잇는 하이퍼 링크들의 리스트 리턴
    #The purpose of Step 3 is to retrieve the hyperlinks from a given URL (url) that are within the same domain as the specified local_domain. It filters out hyperlinks that belong to external domains and focuses only on the hyperlinks within the desired domain.


################################################################################
### Step 4
################################################################################
#도메인 내의 웹페이지를(URL)을 크로울링 하고 텍스트 내용을 저장
def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files     #텍스트 파일 디렉토리
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + local_domain + "/"):
        os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files    #csv 파일 디렉토리
    if not os.path.exists("processed"):
        os.mkdir("processed")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url)  # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open('text/' + local_domain + '/' + url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            # Get the text but remove the tags
            text = soup.get_text()

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")

            # Otherwise, write the text to the file in the text directory
            f.write(text)
        #하이퍼 링크 리스트를 불러오는 함수를 통해 다음 for in loop 실행
        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)


crawl(full_url)


################################################################################
### Step 5
################################################################################
#줄바꿈 제거
# 텍스트를 클린업 하는 과정 (더 순조로운 organizing 을 위해)
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


################################################################################
### Step 6
################################################################################
#텍스트 파일을 수집하고 그 텍스트 파일로 부터 df 만든후 그곳에 저장
# Create a list to store the text files
texts = []
#url을 파싱해서 데이타 프레임에 저장할 경우 사용하는 코드
# Get all the text files in the text directory
for file in os.listdir("text/" + domain + "/"):
    # Open the file and read the text
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns=['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

# 밑 코드는 특정 txt파일만을 df로 저장할때 유동적으로 사용하면됨
# 이전체 코드를 적을경우 step4까지 필요없음 삭제 시켜도됨
"""

def create_dataframe_from_text_file(text_file_path):
    # Read the contents of the text file
    with open(text_file_path, 'r') as file:
        text = file.read()

    # Create a DataFrame with the text content
    df = pd.DataFrame({'text': [text]})

    return df


# Example usage
text_file_path = 'path/to/your/text/file.txt'
df = create_dataframe_from_text_file(text_file_path)


"""
################################################################################
### Step 7
################################################################################
#텍스트 (df) 토큰화 하기
# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# 토큰 분포 시각화
# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

################################################################################
### Step 8
################################################################################
#최대 토큰수 설정하기
max_tokens = 500
# 토큰 수가 최대토큰 수 500 을 넘을경우 텍스트를 청크로 분할

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks


shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['text'])

################################################################################
### Step 9
################################################################################
# 짧아진 토큰수와 토큰을 합쳐서 df생성하기  (그냥 df 에서 토크나이즈 프로세스를 통해 최대 토큰수를 넘지않게 조절후 다시 한번 df 에 저장)
df = pd.DataFrame(shortened, columns=['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()

################################################################################
### Step 10
################################################################################
# 임베딩하기 (api이용) : 저렇게 분할된 토큰(텍스트)으로 이루어진 df 를 임베딩하는 것임
# 임베딩 프로세스 입니다
# Note that you may run into rate limit issues depending on how many files you try to embed
# Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits

df['embeddings'] = df.text.apply(
    lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()

################################################################################
### Step 11
################################################################################
#질문에 대한 문맥 생성 및 DataFrame에서 가장 유사한 문맥 찾기
df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array) #numerical exoression

df.head()


################################################################################
### Step 12
################################################################################
#가장 유사한 문맥을 기반으로 질문에 대답하기
def create_context(
        question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        model="text-davinci-003",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=150,# 스텝 8 의 맥스 토큰량과 다름
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


################################################################################
### Step 13
################################################################################

if __name__ == '__main__':
#예시 질문-응답 호출 제공하기
    print(answer_question(df, question="What day is it?", debug=False))     #not creating debugging information

    print(answer_question(df, question="What is our newest embeddings model?"))