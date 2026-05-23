-- Lists all genres not linked to the show Dexter
-- The tv_shows table contains only one record where title = Dexter
-- Results must be sorted in ascending order by the genre name
SELECT name
FROM tv_genres
WHERE id NOT IN (
    SELECT genre_id
    FROM tv_show_genres
    INNER JOIN tv_shows ON tv_show_genres.show_id = tv_shows.id
    WHERE tv_shows.title = 'Dexter'
)
ORDER BY name ASC;
