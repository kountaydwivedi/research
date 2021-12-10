---
layout: page
title: Finder
permalink: /finder/
order: 1
---

<html>
<body>
<style>
input[type=text], select {
  width: 100%;
  padding: 12px 20px;
  margin: 8px 0;
  display: inline-block;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-sizing: border-box;
}

 {
  border-radius: 5px;
  background-color: #f2f2f2;
  padding: 20px;
}
</style>

<h3>Find your post</h3>

<div id="search-container">
    <input type="text" id="search-input" placeholder="Keywords, Hashtags, anything would work :) ">
	<ul id="results-container"></ul>
</div>

<!-- Script pointing to search-script.js -->
<script src="/research/js/search-script.js" type="text/javascript"></script>

<!-- Configuration -->
<script>
SimpleJekyllSearch({
  searchInput: document.getElementById('search-input'),
  resultsContainer: document.getElementById('results-container'),
  json: '/research/search.json'
})
</script>

</body>
</html>


