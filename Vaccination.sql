use Vaccination
go
Select * from dbo.country_vaccination_stats
Select * from dbo.country_vaccination_stats where country = 'Kuwait'
go
UPDATE vr
SET daily_vaccinations = COALESCE(dt.median_daily_vaccinations, 0)
FROM dbo.country_vaccination_stats vr
LEFT JOIN (
    SELECT country, ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY daily_vaccinations) OVER (PARTITION BY country), 0) AS median_daily_vaccinations
    FROM dbo.country_vaccination_stats
    WHERE daily_vaccinations IS NOT NULL
) AS dt ON vr.country = dt.country
WHERE vr.daily_vaccinations IS NULL;













