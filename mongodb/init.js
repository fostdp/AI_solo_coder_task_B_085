conn = new Mongo();
db = conn.getDB("jade_monitor");

db.createCollection("jade_artifacts");
db.jade_artifacts.createIndex({ artifact_id: 1 }, { unique: true });
db.jade_artifacts.createIndex({ culture: 1 });

/* ======= 出土坑位 GeoJSON 索引（修复：地理位置查询慢） ======= */
db.jade_artifacts.createIndex(
    { "location.pit": "2dsphere" },
    { name: "location.pit_2dsphere", background: true, sparse: true }
);
db.jade_artifacts.createIndex(
    { culture: 1, "location.pit": "2dsphere" },
    {
        name: "culture_1_location.pit_2dsphere",
        background: true,
        partialFilterExpression: { "location.pit": { $exists: true } }
    }
);
db.jade_artifacts.createIndex(
    { "location.site_name": 1, "location.pit_number": 1 },
    { name: "site_pit_compound", background: true }
);

db.createCollection("spectrum_data");
db.spectrum_data.createIndex({ artifact_id: 1, timestamp: -1 });
db.spectrum_data.createIndex({ device_id: 1 });
db.spectrum_data.createIndex(
    { timestamp: 1, artifact_id: 1 },
    { name: "timestamp_1_artifact_id_1", background: true }
);

db.createCollection("raman_spectrum");
db.raman_spectrum.createIndex({ artifact_id: 1, timestamp: -1 });

db.createCollection("xrf_spectrum");
db.xrf_spectrum.createIndex({ artifact_id: 1, timestamp: -1 });

db.createCollection("diffusion_results");
db.diffusion_results.createIndex({ artifact_id: 1, timestamp: -1 });

db.createCollection("anomaly_results");
db.anomaly_results.createIndex({ artifact_id: 1, timestamp: -1 });

db.createCollection("alerts");
db.alerts.createIndex({ artifact_id: 1, timestamp: -1 });
db.alerts.createIndex({ status: 1 });
db.alerts.createIndex(
    { status: 1, timestamp: -1 },
    { name: "status_timestamp", background: true }
);

db.createCollection("devices");
db.devices.createIndex({ device_id: 1 }, { unique: true });

var cultures = ["红山文化", "良渚文化"];
var jadeTypes = ["玉璧", "玉琮", "玉钺", "玉璜", "玉珠", "玉管", "玉兽", "玉鸟"];
var images = [
    "jade_bi.png", "jade_cong.png", "jade_yue.png", "jade_huang.png",
    "jade_zhu.png", "jade_guan.png", "jade_shou.png", "jade_niao.png"
];

/* 考古遗址标准坐标 (WGS84: 经度, 纬度) */
var siteCoordinates = {
    /* 红山文化遗址群 */
    "牛河梁遗址": { lng: 119.3870, lat: 41.3190, radius: 0.05 },
    "那斯台遗址": { lng: 118.9390, lat: 42.2610, radius: 0.04 },
    "兴隆洼遗址": { lng: 120.0780, lat: 42.2750, radius: 0.04 },
    /* 良渚文化遗址群 */
    "反山遗址":   { lng: 120.0000, lat: 30.3950, radius: 0.05 },
    "莫角山遗址": { lng: 119.9930, lat: 30.4010, radius: 0.04 },
    "瑶山遗址":   { lng: 120.0010, lat: 30.4100, radius: 0.04 }
};

/* 红山/良渚对应的遗址备选列表 */
var hongshanSites = ["牛河梁遗址", "那斯台遗址", "兴隆洼遗址"];
var liangzhuSites = ["反山遗址", "莫角山遗址", "瑶山遗址"];

for (var i = 1; i <= 200; i++) {
    var culture = cultures[i % 2];
    var jadeType = jadeTypes[i % jadeTypes.length];
    var image = images[i % images.length];
    var isForgery = Math.random() < 0.15;

    /* 根据文化选择遗址 */
    var siteList = culture === "红山文化" ? hongshanSites : liangzhuSites;
    var excavationSite = siteList[Math.floor(Math.random() * siteList.length)];
    var site = siteCoordinates[excavationSite];

    /* 在遗址中心附近随机模拟坑位坐标 */
    var angle = (i * 0.6180339887 + Math.random()) * Math.PI * 2;
    var distance = Math.sqrt(Math.random() * 0.8) * site.radius;
    var deltaLng = distance * Math.cos(angle) / Math.cos(site.lat * Math.PI / 180);
    var deltaLat = distance * Math.sin(angle);
    var pitLng = site.lng + deltaLng;
    var pitLat = site.lat + deltaLat;

    var pitPrefix = culture === "红山文化" ? "N" : "M";
    var pitNumber = pitPrefix + ((i - 1) % 25 + 1);

    db.jade_artifacts.insertOne({
        artifact_id: "JD" + String(i).padStart(4, "0"),
        name: culture + jadeType + "-" + i,
        culture: culture,
        jade_type: jadeType,
        image_file: image,
        excavation_site: excavationSite,
        excavation_year: 2010 + (i % 15),
        description: "出土于" + (culture === "红山文化" ? "辽宁朝阳" : "浙江余杭") + "的典型" + jadeType,
        /* ======== GeoJSON 出土坑位位置 ======== */
        location: {
            pit: {
                type: "Point",
                coordinates: [ pitLng, pitLat ]
            },
            site_name: excavationSite,
            pit_number: pitNumber,
            layer_depth_m: Number((0.5 + Math.random() * 4.5).toFixed(2)),
            grid_cell: {
                zone: Math.floor((i - 1) / 20) + 1,
                row: String.fromCharCode(65 + ((i - 1) % 10)),
                column: ((i - 1) % 20) + 1
            },
            region: culture === "红山文化"
                ? { province: "辽宁", city: "朝阳", district: "凌源市" }
                : { province: "浙江", city: "杭州", district: "余杭区" }
        },
        /* ======== 玉质结构（用于各向异性张量标定） ======== */
        texture: {
            ct_scan_id: "CT-" + culture.substring(0, 2) + "-" + String(i).padStart(4, "0"),
            main_orientation_euler: [
                Number((Math.random() * 180 - 90).toFixed(2)),
                Number((Math.random() * 90).toFixed(2)),
                Number((Math.random() * 180).toFixed(2))
            ],
            grain_size_um: Number((30 + Math.random() * 80).toFixed(1)),
            porosity: Number((0.01 + Math.random() * 0.04).toFixed(4))
        },
        size: {
            length: Number((5 + Math.random() * 15).toFixed(2)),
            width: Number((3 + Math.random() * 10).toFixed(2)),
            thickness: Number((0.5 + Math.random() * 2).toFixed(2))
        },
        weight: Number((50 + Math.random() * 500).toFixed(2)),
        is_suspected_forgery: isForgery,
        create_time: new Date(),
        update_time: new Date()
    });
}

for (var i = 1; i <= 20; i++) {
    db.devices.insertOne({
        device_id: "RAMAN" + String(i).padStart(3, "0"),
        device_type: "raman",
        model: "Renishaw-inVia",
        status: "online",
        location: "实验室A区-" + i,
        last_heartbeat: new Date()
    });
}

for (var i = 1; i <= 20; i++) {
    db.devices.insertOne({
        device_id: "XRF" + String(i).padStart(3, "0"),
        device_type: "xrf",
        model: "Bruker-S8",
        status: "online",
        location: "实验室B区-" + i,
        last_heartbeat: new Date()
    });
}

var artifactCount = db.jade_artifacts.countDocuments();
var spatialCount = db.jade_artifacts.countDocuments({ "location.pit": { $exists: true } });

print("数据库初始化完成：" + artifactCount + "件玉器，40台设备");
print("  已建立 location.pit (2dsphere) 空间索引");
print("  带有GeoJSON坑位数据的玉器数: " + spatialCount);
