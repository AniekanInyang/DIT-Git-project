select description, sum(quantity) as total_quantity 
from e_commerce
group by description
order by total_quantity desc
limit 1;
