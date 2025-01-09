from ingnore import create_dataset
from database import upload_json_data, create_collection

upload_json_data(
        'social01',
        "social_media_data.json", 
        lambda data: (
            f"hashtags: {', '.join(data['hashtags'])} | "
            f"post_type: {data['post_type']} | "
            f"total_impressions: {data['total_impressions']} | "
            f"comments: {data['comments']} | "
            f"shares: {data['shares']} | "
            f"media_quality: {data['media_quality']} | "
            f"audience: {data['audience_demographic']['age']} {data['audience_demographic']['gender']} {data['audience_demographic']['location']} | "
            f"conversion_rate: {data['conversion_rate']}"
        ),
    )