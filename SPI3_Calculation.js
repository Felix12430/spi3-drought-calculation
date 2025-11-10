/**
 * SPI-3 Calculation and Training Points Extraction
 * Computes 3-month Standardized Precipitation Index from CHIRPS data
 * Generates training points for Random Forest classification
 * Author: [FELIX WAIGIRI KIRUKI]
 * Date: [10/11/2025]
 */
// Load CHIRPS Daily Rainfall Data
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
               .filterDate('2005-01-01', '2024-12-31')
               .select('precipitation');
// Function to compute 3-month rolling sum
var compute3MonthSum = function(date) {
  var start = ee.Date(date).advance(-2, 'month'); // Go back 2 months
  var end = ee.Date(date).advance(1, 'day'); // Include current month
  var sum = chirps.filterDate(start, end).sum(); // Sum precipitation over 3 months
  return sum.set('system:time_start', date);
};
// Generate list of monthly timestamps
var months = ee.List.sequence(0, ee.Date('2024-12-01').difference(ee.Date('2005-01-01'), 'month'))
  .map(function(n) { return ee.Date('2005-01-01').advance(n, 'month'); });
// Compute 3-month summed images
var chirps_3month = ee.ImageCollection(months.map(compute3MonthSum));
// Compute the mean and standard deviation over the 3-month summed rainfall
var meanPrecip = chirps_3month.mean().rename('precipitation');  
var stdPrecip = chirps_3month.reduce(ee.Reducer.stdDev()).rename('precipitation');
var maxPrecip = chirps_3month.max().rename('max_precipitation');
var minPrecip = chirps_3month.min().rename('min_precipitation');
// Define the study area
var studyArea = marsabit.geometry();
// Reduce images to get numerical values
var meanStats = meanPrecip.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: studyArea,
  scale: 1000,
  bestEffort: true
});
var stdStats = stdPrecip.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: studyArea,
  scale: 1000,
  bestEffort: true
});
var maxStats = maxPrecip.reduceRegion({
  reducer: ee.Reducer.max(),
  geometry: studyArea,
  scale: 1000,
  bestEffort: true
});
var minStats = minPrecip.reduceRegion({
  reducer: ee.Reducer.min(),
  geometry: studyArea,
  scale: 1000,
  bestEffort: true
});
var Stats = minPrecip.reduceRegion({
  reducer: ee.Reducer.min(),
  geometry: studyArea,
  scale: 1000,
  bestEffort: true
});
// Print actual values
print("Mean 3-month Precipitation:", meanStats.get('precipitation'));
print("Standard Deviation of 3-month Precipitation:", stdStats.get('precipitation'));
print("Maximum 3-month Precipitation:", maxStats.get('max_precipitation'));
print("Minimum 3-month Precipitation:", minStats.get('min_precipitation'));
// Compute SPI-3
var spi3Collection = chirps_3month.map(function(image) {
  return image.subtract(meanPrecip)
              .divide(stdPrecip.where(stdPrecip.eq(0), 0.001))
              .rename('SPI-3')
              .set('system:time_start', image.get('system:time_start'));
});
// Define drought periods
var droughtPeriods = [
  {name: '2005 Drought', start: '2004-06-01', end: '2006-01-31'},
  {name: '2010-2011 Drought', start: '2010-06-01', end: '2011-12-31'},
  {name: '2016-2017 Drought', start: '2016-06-01', end: '2017-12-31'},
  {name: '2020-2022 Drought', start: '2020-06-01', end: '2022-12-31'}
];
// Function to compute SPI-3 statistics for each drought period
droughtPeriods.forEach(function(period) {
  var spiPeriod = spi3Collection.filterDate(period.start, period.end).mean();
  // Calculate statistics within the study area
  var stats = spiPeriod.reduceRegion({
    reducer: ee.Reducer.min().combine({
      reducer2: ee.Reducer.max(),
      sharedInputs: true
    }).combine({
      reducer2: ee.Reducer.mean(),
      sharedInputs: true
    }),
    geometry: studyArea,
    scale: 1000,
    bestEffort: true
  });
  // Print results
  print(period.name, stats);
});
// Function to filter SPI-3 collection based on the drought periods
var filteredCollections = droughtPeriods.map(function(period) {
  return spi3Collection.filterDate(period.start, period.end);
});
// Merge all filtered collections into a single ImageCollection
var combinedSPI3 = ee.ImageCollection(
  filteredCollections.reduce(function(col1, col2) {
    return col1.merge(col2);
  })
).mean().clip(studyArea);
// Define SPI-3 drought thresholds based on past observations
var droughtClasses = [
  {min: -0.4, max: -0.3, label: 'Severe to Extreme Drought', class: 0}, 
  {min: -0.3, max: -0.2, label: 'Moderate Drought', class: 1}, 
  {min: -0.2, max: 0.0, label: 'Mild Drought', class: 2}, 
  {min: 0.0, max: 0.06, label: 'Near Normal Conditions', class: 3}
];
// Generate training points with improved sampling strategy
var numSamplesPerClass = 200; // Ensure balance across classes
// Initialize empty FeatureCollection for training points
var trainingPoints = ee.FeatureCollection([]);
// Generate training points for each drought class
droughtClasses.forEach(function(threshold) {
  var classifiedPoints = combinedSPI3.updateMask(
      combinedSPI3.gte(threshold.min).and(combinedSPI3.lt(threshold.max)))
      .sample({
        region: studyArea,
        scale: 1000,
        numPixels: numSamplesPerClass,
        geometries: true 
      }).map(function(f) { 
        return f.set('class', threshold.class);
      });
  trainingPoints = trainingPoints.merge(classifiedPoints);
});
// Print total training points for verification
print("Total Training Points:", trainingPoints.size());
// Display training points on the map
Map.centerObject(studyArea, 8);
Map.addLayer(trainingPoints, {color: 'red'}, 'Training Points');
print("SPI-3 Statistics:",combinedSPI3.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: studyArea,
  scale: 1000,
  bestEffort: true
}));
// Count training points per drought class
droughtClasses.forEach(function(threshold) {
  var classPoints = trainingPoints.filter(ee.Filter.eq('class', threshold.class));
  print("Class " + threshold.class + " (" + threshold.label + ") Points:", classPoints.size());
});
// Step 4: Reduce SPI to extract Max & Min SPI values over time
var maxSPI = spi3Collection.max().rename('Max_SPI_3');
var minSPI = spi3Collection.min().rename('Min_SPI_3');
// Step 5: Print Max and Min SPI values
var studyArea = marsabit.geometry(); 
var maxSPIValue = maxSPI.reduceRegion({
  reducer: ee.Reducer.max(),
  geometry: marsabit,
  scale: 1000,
  bestEffort: true
});
var minSPIValue = minSPI.reduceRegion({
  reducer: ee.Reducer.min(),
  geometry: marsabit,
  scale: 1000,
  bestEffort: true
});
print("Maximum SPI-3 Value:", maxSPIValue.get('Max_SPI_3'));
print("Minimum SPI-3 Value:", minSPIValue.get('Min_SPI_3'));
//  Plot Time Series Chart of SPI-3
var chart = ui.Chart.image.series({
  imageCollection: spi3Collection,
  region: marsabit,
  reducer: ee.Reducer.mean(),
  scale: 1000,
  xProperty: 'system:time_start'
})
.setOptions({
  title: 'SPI-3 Time Series (2005-2024)',
  hAxis: {title: 'Year'},
  vAxis: {title: 'SPI-3 Value'},
  series: {0: {color: 'blue'}}
});
print(chart);
var totalImages = spi3Collection.size().getInfo(); // Get total image count
print('Total SPI-3 Images:', totalImages);
// Define SPI-3 visualization parameters
var visParams = {
  min: -0.4, 
  max: 0.1, 
  palette: ['red', 'orange', 'yellow', 'lightgreen', 'green']
};
// Add SPI-3 layer to the map
Map.centerObject(studyArea, 8);
Map.addLayer(combinedSPI3, visParams, 'SPI-3 Composite');
// Convert trainingPoints back to FeatureCollection (Fix for empty collection issue)
trainingPoints = ee.FeatureCollection(trainingPoints);
// Extract Longitude, Latitude, SPI-3 Value, and Class for CSV
var formattedPoints = trainingPoints.map(function(f) {
  var coords = f.geometry().coordinates();
  return ee.Feature(f.geometry(), {
    'longitude': coords.get(0),
    'latitude': coords.get(1),
    'SPI-3': f.get('SPI-3'),
    'class': f.get('class')
  });
});
// Export Training Points as CSV (Longitude, Latitude, Class, SPI-3)
Export.table.toDrive({
  collection: formattedPoints,
  description: 'SPI3_Training_Clipped',
  fileFormat: 'CSV'
});
// Export Clipped SPI-3 as GeoTIFF
Export.image.toDrive({
  image: combinedSPI3,
  description: 'SPI3_Drought_2005_2010_2011_Clipped',
  scale: 1000,
  region: studyArea, 
  fileFormat: 'GeoTIFF',
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
