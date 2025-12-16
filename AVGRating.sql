select m.food_id,m.food_name,SUM(r.rating)/COUNT(1) as rating
from Menus m
left join
	Ratings r on r.food_id = m.food_id
group by
	m.food_id,m.food_name
order by
	m.food_id asc

select * from Menus left join Ratings on Menus.food_id = Ratings.food_id
where Menus.food_id = 5