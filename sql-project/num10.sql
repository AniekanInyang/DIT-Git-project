select customer_id, sum(quantity) as total_quantity, sum(unit_price) as total_unit_price
from e_commerce
group by customer_id
order by total_unit_price desc;