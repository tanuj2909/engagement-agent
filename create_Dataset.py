from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from pydantic import BaseModel, Field, model_validator
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
    location: str = Field(description="The country of the audience.", example="USA")

class SocialMediaPost(BaseModel):
    hashtags: list[str] = Field(
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
    conversion_rate: float = Field(
        description="The conversion rate of the post as a percentage (e.g., 0.1 to 10%).",
        example=4.5
    )

    # You can add custom validation logic easily with Pydantic.
    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, values: dict) -> dict:
        setup = values.get("setup")
        if setup and setup[-1] != "?":
            raise ValueError("Badly formed question!")
        return values
    
class mediaPost(BaseModel):
    model: list[SocialMediaPost] = Field(description="nothing")



def clean_output(output: str) -> str:
    json_match = re.search(r"\[.*\]", output, re.DOTALL)
    if json_match:
        return json_match.group(0)
    raise ValueError("Invalid JSON format detected in model output.")



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
    Location: Choose from [USA, India, UK, Germany, Brazil, Australia, Canada, France, Japan, South Korea, South Africa, Italy, Spain, Russia, Mexico, Netherlands, China, Sweden, New Zealand, UAE].
    Younger audiences (18-24) are more likely to comment, while older audiences (25-40) are more likely to share.
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
    parser = PydanticOutputParser(pydantic_object=mediaPost)

    prompt = PromptTemplate(
        input_variables=[],  # No variables since the prompt is static
        template=prompt,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # database = connect_to_database()

    # collection = create_collection(database, "Media Engagement")

    Answer = []

    prompt_and_model = prompt | model 
    # And a query intended to prompt a language model to populate the data structure.
    for x in range (1, 100):
        output = prompt_and_model.invoke({})
        results = parser.invoke(output)

        for result in results:
            Answer.append(result)

    for x in range(1, 5):  # Test with a smaller range first
        try:
            raw_output = prompt_and_model.invoke({})
            clean_json = clean_output(raw_output)  # Extract and clean JSON
            results = parser.parse_result(clean_json)  # Parse the cleaned JSON
            Answer.extend(results)
        except Exception as e:
            print(f"Error processing iteration {x}: {e}")

        
    output_file = "social_media_data.json"
    with open(output_file, "w") as json_file:
        json.dump(Answer, json_file, indent=4)


    # upload_json_data(
    #     collection,
    #     "social_media_data.json", 
    #     lambda data: (
    #         f"hashtags: {', '.join(data['hashtags'])} | "
    #         f"post_type: {data['post_type']} | "
    #         f"total_impressions: {data['total_impressions']} | "
    #         f"comments: {data['comments']} | "
    #         f"shares: {data['shares']} | "
    #         f"media_quality: {data['media_quality']} | "
    #         f"audience: {data['audience_demographic']['age']} {data['audience_demographic']['gender']} {data['audience_demographic']['location']} | "
    #         f"conversion_rate: {data['conversion_rate']}"
    #     ),
    # )