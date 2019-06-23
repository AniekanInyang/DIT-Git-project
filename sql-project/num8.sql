select invoice_no, sum(quantity*unit_price)amount
from e_commerce
where invoice_no = '536370'
group by invoice_no;