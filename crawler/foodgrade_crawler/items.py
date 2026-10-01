import scrapy

class FoodProductItem(scrapy.Item):
    url = scrapy.Field()
    page_title = scrapy.Field()
    product_name = scrapy.Field()
    brand = scrapy.Field()
    category = scrapy.Field()
    ingredients = scrapy.Field()
    claims = scrapy.Field()
    source = scrapy.Field()
