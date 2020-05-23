#include "Config.hpp"

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <stdexcept>

namespace tppocr {

void Config::parseFromTOML(const std::string path) {
    toml::table table;

    try {
        table = toml::parse_file(path);
    } catch (const toml::parse_error & error) {
        std::cerr
            << "Error parsing file '" << *error.source().path
            << "':\n" << error.description()
            << "\n  (" << error.source().begin << ")"
            << std::endl;
    }

    realTime = getTOMLNode(table, "realtime").as_boolean()->get();
    processingFPS = getTOMLNode(table, "processing-fps").as_floating_point()->get();
    tessdataPath = getTOMLNode(table, "tessdata").as_string()->get();
    detectorModelPath = getTOMLNode(table, "detector-model").as_string()->get();
    detectorConfidenceThreshold = getTOMLNode(table, "detector-confidence-threshold").as_floating_point()->get();
    detectorNonmaximumSuppressionThreshold = getTOMLNode(table, "detector-nonmaximum-suppression-threshold").as_floating_point()->get();
    recognizerConfidenceThreshold = getTOMLNode(table, "recognizer-confidence-threshold").as_floating_point()->get();

    for (const auto & node : *table["region"].as_array()) {
        const auto & regionConfig = *node.as_table();
        Region region;
        region.name = regionConfig["name"].as_string()->get();
        region.x = regionConfig["x"].as_integer()->get();
        region.y = regionConfig["y"].as_integer()->get();
        region.width = regionConfig["width"].as_integer()->get();
        region.height = regionConfig["height"].as_integer()->get();

        region.alwaysHasText = regionConfig["always-has-text"].value_or<bool>(false);
        region.patternFilename = regionConfig["recognizer-pattern-file"].value_or<std::string>("");

        regions.push_back(region);

        std::cerr << "Configured region '" << region.name << "'" << std::endl;
    }
}

toml::node_view<toml::node> Config::getTOMLNode(toml::table & table, const std::string key) {
    auto view = table[key];

    if (view) {
        return view;
    } else {
        throw std::runtime_error("Missing key" + key);
    }
}

}
