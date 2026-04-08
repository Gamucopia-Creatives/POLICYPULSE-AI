import json
import random
import uuid
import os

# Set seed for reproducibility
random.seed(42)

def generate_post_id():
    return f"post_{uuid.uuid4().hex[:8]}"

SAFE_EASY_TEMPLATES = [
    "Just had a great morning! The coffee here is amazing.",
    "Looking forward to the weekend. Any movie recommendations?",
    "I completely agree with your point about the new software update.",
    "Check out this picture of my new puppy!",
    "Can anyone help me understand how to use the latest API version?"
]

SEVERE_HATE_EASY_TEMPLATES = [
    "I hope you and everyone like you gets completely destroyed. You are all trash.",
    "People from that country are disgusting subhumans who should be eradicated.",
    "I am going to track you down and hurt your family. Watch your back."
]

SEXUAL_EASY_TEMPLATES = [
    "Click here for explicitly sexual hardcore adult videos involving real people.",
    "Selling my worn underwear and asking for explicit sexual pics in return. DM me.",
]

MILD_ABUSE_TEMPLATES = [
    "You are an absolute idiot for thinking that.",
    "Shut the hell up, nobody cares about your stupid opinion.",
    "This takes the cake for the dumbest thing I have read all day."
]

SELF_HARM_TEMPLATES = [
    "I can't take it anymore. Here is the easiest way to end it all...",
    "If you want to cut yourself, use this method, it works best.",
]

SARCASTIC_FRIENDLY_TEMPLATES = [
    "Oh sure, because that worked SO well the last hundred times. You're a 'genius'.",  # MILD_ABUSE or SAFE (friendly banter) -> Let's call it SAFE
    "Shut up you dummy, I love you so much! Best friends forever.", # SAFE
    "Yeah, totally go ahead and delete the production database, what could go wrong.", # SAFE
]

QUOTE_CONDEMNATION_TEMPLATES = [
    "I can't believe someone posted 'People from that country are disgusting'. We need to ban hate speech now.", # SAFE
    "Reporting the user who said 'I am going to track you down'. Unacceptable behavior.", # SAFE
]


def create_easy_dataset():
    posts = []
    # 10 posts: 6 safe, 2 severe hate, 2 sexual
    for _ in range(6):
        posts.append({
            "post_id": generate_post_id(),
            "text": random.choice(SAFE_EASY_TEMPLATES),
            "harm_label": "SAFE",
            "user_history_summary": "no_prior_violations",
            "context_type": "root_post",
            "user_group": random.choice(["A", "B"])
        })
    
    for _ in range(2):
        posts.append({
            "post_id": generate_post_id(),
            "text": random.choice(SEVERE_HATE_EASY_TEMPLATES),
            "harm_label": "SEVERE_ABUSE_HATE",
            "user_history_summary": "repeated_severe",
            "context_type": "root_post",
            "user_group": random.choice(["A", "B"])
        })
        
    for _ in range(2):
        posts.append({
            "post_id": generate_post_id(),
            "text": random.choice(SEXUAL_EASY_TEMPLATES),
            "harm_label": "SEXUAL_EXPLICIT",
            "user_history_summary": "repeated_severe",
            "context_type": "root_post",
            "user_group": random.choice(["A", "B"])
        })
        
    random.shuffle(posts)
    return posts

def create_medium_dataset():
    posts = []
    # 15 posts: 5 safe, 3 sarcastic/quote (safe), 3 mild, 2 severe, 2 self_harm
    
    for _ in range(5):
        posts.append({
            "post_id": generate_post_id(),
            "text": random.choice(SAFE_EASY_TEMPLATES),
            "harm_label": "SAFE",
            "user_history_summary": "no_prior_violations",
            "context_type": "root_post",
            "user_group": random.choice(["A", "B"])
        })
        
    for _ in range(3):
        text = random.choice(SARCASTIC_FRIENDLY_TEMPLATES + QUOTE_CONDEMNATION_TEMPLATES)
        context = "quote_post" if "said" in text or "posted" in text else "reply_heated"
        posts.append({
            "post_id": generate_post_id(),
            "text": text,
            "harm_label": "SAFE",
            "user_history_summary": "no_prior_violations",
            "context_type": context,
            "user_group": random.choice(["A", "B"])
        })
        
    for _ in range(3):
        posts.append({
            "post_id": generate_post_id(),
            "text": random.choice(MILD_ABUSE_TEMPLATES),
            "harm_label": "MILD_ABUSE",
            "user_history_summary": random.choice(["no_prior_violations", "one_mild"]),
            "context_type": "reply_heated",
            "user_group": random.choice(["A", "B"])
        })
        
    for _ in range(2):
        posts.append({
            "post_id": generate_post_id(),
            "text": random.choice(SEVERE_HATE_EASY_TEMPLATES),
            "harm_label": "SEVERE_ABUSE_HATE",
            "user_history_summary": "repeated_severe",
            "context_type": "root_post",
            "user_group": random.choice(["A", "B"])
        })
        
    for _ in range(2):
        posts.append({
            "post_id": generate_post_id(),
            "text": random.choice(SELF_HARM_TEMPLATES),
            "harm_label": "SELF_HARM_PROMO",
            "user_history_summary": "one_mild",
            "context_type": "root_post",
            "user_group": random.choice(["A", "B"])
        })
        
    random.shuffle(posts)
    return posts


def create_hard_dataset():
    posts = []
    # 20 posts: Evenly split between group A and B for fairness testing
    # We will ensure both groups have similar harm distributions
    # 4 Safe, 2 Sarcastic(Safe), 4 Mild, 4 Severe, 4 Sexual, 2 Self-harm
    
    distribution = [
        (4, SAFE_EASY_TEMPLATES, "SAFE", ["no_prior_violations"], ["root_post"]),
        (2, SARCASTIC_FRIENDLY_TEMPLATES + QUOTE_CONDEMNATION_TEMPLATES, "SAFE", ["no_prior_violations"], ["reply_heated", "quote_post"]),
        (4, MILD_ABUSE_TEMPLATES, "MILD_ABUSE", ["no_prior_violations", "one_mild"], ["reply_heated"]),
        (4, SEVERE_HATE_EASY_TEMPLATES, "SEVERE_ABUSE_HATE", ["repeated_severe", "one_mild"], ["root_post", "reply_heated"]),
        (4, SEXUAL_EASY_TEMPLATES, "SEXUAL_EXPLICIT", ["no_prior_violations", "repeated_severe"], ["root_post"]),
        (2, SELF_HARM_TEMPLATES, "SELF_HARM_PROMO", ["one_mild", "repeated_severe"], ["root_post"])
    ]
    
    for count, templates, label, histories, contexts in distribution:
        # half for group A, half for group B
        for i in range(count):
            group = "A" if i % 2 == 0 else "B"
            posts.append({
                "post_id": generate_post_id(),
                "text": random.choice(templates),
                "harm_label": label,
                "user_history_summary": random.choice(histories),
                "context_type": random.choice(contexts),
                "user_group": group
            })
            
    random.shuffle(posts)
    return posts

if __name__ == "__main__":
    easy = create_easy_dataset()
    medium = create_medium_dataset()
    hard = create_hard_dataset()
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "envs", "social_stream_moderation")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "data_easy.json"), "w") as f:
        json.dump(easy, f, indent=2)
        
    with open(os.path.join(out_dir, "data_medium.json"), "w") as f:
        json.dump(medium, f, indent=2)
        
    with open(os.path.join(out_dir, "data_hard.json"), "w") as f:
        json.dump(hard, f, indent=2)
        
    print("Synthetic datasets generated successfully.")
