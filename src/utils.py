from nltk.tokenize import word_tokenize
import constants

def get_labels_from_post_list(post_list):
  labels = []
  for post in post_list:
    received = post["requester_received_pizza"]
    if received:
      labels.append(1)
    else: 
      labels.append(0)
  return labels


def tokenize(string):
  # can add more here
  # ngrams too
  return word_tokenize(string) 

def get_post_tokens(post):
  # get text  
  body_text = post["request_text_edit_aware"]
  title_text = post["request_title"]
  # tokenize
  body_tokens = tokenize(body_text)  
  title_tokens =tokenize(title_text)
  return body_tokens, title_tokens

def get_flair_index(flair):
  return constants.FLAIR_INDICES[flair] 


def write_output(data, labels):
  if len(data) != len(labels):
    print 'SOMETHING IS FUCKY'
  with open('../data/output.csv', 'w') as output:
    output.write('request_id,requester_received_pizza\n')
    for i in range(len(data)):
      post = data[i]
      label = labels[i]
      request_id = post["request_id"]
      output.write(request_id + ',' + str(label) + '\n')

