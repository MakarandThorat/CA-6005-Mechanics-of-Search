<!DOCTYPE html>
<html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Search Results</title>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
            <script>
                $(document).ready(function(){
                    $('body').css("margin", "20px");
                    $('body').css("color", "#181818");
                    $('body').css("padding", "20px");
                    $('body').css("font-family", "Poppins");
                    $("head").append(`<link rel="preconnect" href="https://fonts.googleapis.com">
                    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">`);
                });
        
            </script>
            <style>
                img{
                    max-width: 300px;
                    max-height: 300px;
                    border-radius: 10px;
                }
                
            </style>
        </head>
<body>
    <div style='margin-top: 50px; padding: 20px; text-align: center;'>
    <h1>Search Results for "{{ query }}"</h1>
    
    {% if results %}
        {% for result in results %}
        <div>
            <h2>Ranked {{ loop.index }} - Score: {{ result.score }}</h2>
            <img src="{{ url_for('static', filename=result.image_path) }}" alt="Image">
            <p>{{ result.surrogate }}</p>
        </div>
        {% endfor %}
    {% else %}
        <p>No results found.</p>
    {% endif %}
</div>
</body>
</html>
