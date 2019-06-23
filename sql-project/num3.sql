SELECT count (news_article)
from news_article
where extract (year from publish_date) = 2016 and news_type = 'sports' 

