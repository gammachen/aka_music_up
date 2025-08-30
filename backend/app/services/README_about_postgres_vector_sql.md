```sql
SELECT 
    image_path,
    face_path,
    feature_vector <=> '[0.1, 0.2, 0.3, 0.4, 0.5]'::vectors.vector AS cosine_distance,
    1 - (feature_vector <=> '[0.1, 0.2, 0.3, 0.4, 0.5]'::vectors.vector) AS cosine_similarity
FROM face_images
ORDER BY feature_vector <=> '[0.1, 0.2, 0.3, 0.4, 0.5]'::vectors.vector
LIMIT 5;
```


