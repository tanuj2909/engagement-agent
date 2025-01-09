from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from typing import List
from database import connect_to_database, create_collection, upload_json_data
import json
from langchain_mistralai.chat_models import ChatMistralAI
import os
import getpass
from langchain_mistralai import ChatMistralAI
import re

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")


model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
    # other params...
)


# Define your desired data structure.
class AudienceDemographic(BaseModel):
    age: str = Field(description="The age group of the audience (e.g., '18-24').", example="18-24")
    gender: str = Field(description="The gender of the audience (e.g., 'Male', 'Female').", example="Female")
    location: list[str] = Field(description="The country of the audience.", example="USA")

# Define the data model for Social Media Post
class SocialMediaPost(BaseModel):
    hashtags: List[str] = Field(
        description="A list of hashtags associated with the post (1 to 5 hashtags).",
        example=["#trending", "#fitness", "#health"]
    )
    post_type: str = Field(
        description="The type of post, such as 'Carousel', 'Reels', 'Static Image', or 'Video'.",
        example="Reels"
    )
    total_impressions: int = Field(
        description="The total number of impressions the post received.",
        example=850000
    )
    comments: int = Field(
        description="The total number of comments the post received.",
        example=12000
    )
    shares: int = Field(
        description="The total number of shares the post received.",
        example=8000
    )
    media_quality: int = Field(
        description="The quality rating of the post's media on a scale of 1 to 5.",
        example=5
    )
    audience_demographic: AudienceDemographic = Field(
        description="The demographic data of the audience engaging with the post.",
        example={
            "age": "18-24",
            "gender": "Female",
            "location": "USA"
        }
    )

    published_date: int = Field(
        description="how many days ago was this post posted",
        examples=15
    )

    conversion_rate: float = Field(
        description="The conversion rate of the post as a percentage (e.g., 0.1 to 10%).",
        example=4.5
    )

    # Custom validation logic for additional consistency checks
    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, values: dict) -> dict:
        # Add any field-specific or post-specific validation logic here
        return values
    
class SocialMediaPostList(BaseModel):
    posts: list[SocialMediaPost]

    # Custom validation for the list of posts
    @model_validator(mode="before")
    @classmethod
    def validate_posts(cls, values: dict) -> dict:
        posts = values.get("posts", [])
        if not posts:
            raise ValueError("The list of posts cannot be empty.")
        if len(posts) > 100:
            raise ValueError("Too many posts. The maximum allowed is 100.")
        return values

def create_dataset():
    prompt = '''
        You are a data generator for a social media analytics system. Your task is to create a dataset containing a list of dictionaries. Each dictionary should represent a social media post with realistic values for the following fields, starting with Hashtags and Post Type, and then determining the remaining attributes based on their relationships and trends:

    Hashtags:

    A list of 1-5 hashtags selected from a diverse set of categories, such as General Trends, Technology, Lifestyle, Entertainment, Business & Career, Social Issues, or Seasonal hashtags.
    Hashtags should align with the Post Type, Audience Demographic, and Media Quality:
    Reels: Often feature viral or general trend hashtags like #trending or #viral.
    Business-oriented posts: Use hashtags like #startup, #marketing, or #productivity.
    Seasonal posts: Include hashtags such as #christmas during the holiday season.
    Trending or relevant hashtags increase Impressions, Shares, and Conversion Rates.
    Post Type:

    One of the following: "Carousel", "Reels", "Static Image", "Video".
    Reels and Videos typically have higher Engagement Rates (e.g., Comments and Shares) and better Media Quality (3-5).
    Attributes Based on Hashtags and Post Type:
    Total Impressions:

    A numerical value between 500 and 1,000,000.
    Posts with higher Media Quality and trending Hashtags typically have higher Impressions.
    Comments:

    A numerical value between 0 and 50,000.
    Posts with higher Impressions and Media Quality tend to have more Comments.
    Younger audiences (18-24) are more likely to engage with posts, resulting in higher Comments.
    Shares:

    A numerical value between 0 and 100,000.
    Posts with meaningful content or viral Hashtags tend to have more Shares.
    Older audiences (25-40) are more likely to share posts compared to younger audiences.
    MediaQuality:

    A numerical value between 1 and 5 (integer).
    Higher Media Quality increases Impressions and Engagement.
    Reels and Videos generally have better Media Quality (3-5).
    AudienceDemographic:

    A dictionary with fields:
    Age: One of [18-24, 25-34, 35-44, 45-54].
    Gender: One of ["Male", "Female"].
    Location: Choose 2-3 where post went trending from [USA, India, UK, Germany, Brazil, Australia, Canada, France, Japan, South Korea, South Africa, Italy, Spain, Russia, Mexico, Netherlands, China, Sweden, New Zealand, UAE].
    Younger audiences (18-24) are more likely to comment, while older audiences (25-40) are more likely to share.

    Published Time: 
    select number of days to create attribute that can tell how many days ago was this post posted
    
    Conversion Rate:

    A percentage between 0.1% and 10%.
    Posts with higher Impressions and better Media Quality tend to have higher Conversion Rates.
    Generate the Dataset
    Generate a list of 20 dictionaries containing data for the above fields.
    Begin by selecting Hashtags and Post Type, ensuring they align with the intended content and trends.
    Use these selections to determine the remaining attributes, ensuring realistic correlations (e.g., Media Quality impacting Impressions, younger audiences engaging more with Reels, etc.).
    Ensure the data reflects real-life variance and trends in social media post performance.

    {format_instructions}
    '''

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=SocialMediaPostList)

    # Prepare the prompt template with format instructions
    prompt = PromptTemplate(
        input_variables=[],
        template=prompt,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Simulated integration with a language model
    # Replace this section with actual LangChain model setup and invocation
    print("Generating data...")

    # Generate the output dataset
    prompt_and_model = prompt | model  # Assuming 'model' is an LLM instance
    Answer = []
    for x in range(1,20):
        output = prompt_and_model.invoke({})
        # print(output)

        results = parser.invoke(output)
        for result in results['posts']:
            Answer.append(result)
        
    # Parse the output
    # Optionally print or save results
    # print(results)
    
    output_file = "social_media_data.json"
    with open(output_file, "w") as json_file:
        json.dump(Answer, json_file, indent=4)


    