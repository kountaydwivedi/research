---
layout: page
title: Finder
permalink: /finder/
order: 1
---
<div id="search-container">
	<input type="text" id="search-input" placeholder="Search a post...">
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


