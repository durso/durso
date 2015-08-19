var methods = this;

var body = document.body;

/*
 * Should you want to render the json object on the client side, you can use these functions.
 * First, construct a json object on the server-side using depth first traversal, like the one below:
var object = [{"tag":"p","attr":{"class":"arroz"},"hasChild":true,"children":[1,3]},
    {"tag":"p","attr":{"class":"feijao"},"hasChild":true,"children":[2]},
    {"tag":"text","value":"Hello there","hasChild":false},
    {"tag":"text","value":"Hello there again","hasChild":false}];

function elem(tagName){

     this.value;

     this.value = document.createElement(tagName);    

};




elem.prototype.addChild = function(child){

    this.value.appendChild(child);

}

elem.prototype.attr = function(key,value){

    this.value.setAttribute(key,value);

}

elem.prototype.getValue = function(){

    return this.value;

};

function txt(value){

     this.value = document.createTextNode(value);

}

txt.prototype.getValue = function(){

     return this.value;

};

function elemFactory(object){

    this.value;

    var obj = object.shift();

    if(obj.tag === "text"){

        this.value = new txt(obj.value);

    } else {

        this.value = new elem(obj.tag);

        for(var k in obj.attr){
            this.value.attr(k,obj.attr[k]);

        }

        if(obj.hasChild){
   
            for(var i =0; i < obj.children.length; i++){

                this.value.addChild(new elemFactory(object));

               

            }

        }

    }

    return this.value.getValue();

}

Usage:
var result = new elemFactory(object);
body.appendChild(result);

*/

jQuery.fn.extend({

    getText: function () {

        return $(this).contents().filter(function () {

                    return this.nodeType === 3;

                });

    }

});

jQuery.fn.extend({

    removeText: function () {

        jQuery(this).getText().remove();

    }

});
function getChildNode(jq,key){
    var elem = jq[0];
    return elem.childNodes[key];
}
jQuery.fn.extend({

    changeText: function (text, key) {

        var node = getChildNode(jQuery(this),key);

        if(typeof node !== 'undefined'){

            node.nodeValue = text;

        }

    }

});

jQuery.fn.extend({

    appendText: function (text, key) {

        var node = getChildNode(jQuery(this),key);

        if(typeof node !== 'undefined'){
            var tmp = node.nodeValue;
            tmp += text;
            node.nodeValue = tmp;
        }

    }

});







var addClass = function(element,item){

    jQuery(element).addClass(item.value);

}

var attr = function(element,item){

    jQuery(element).attr(item.key,item.value);

}

var removeAttr = function(element,item){

    jQuery(element).removeAttr(item.value);

}



var append = function(element,item){

    jQuery(element).append(item.value);

}

var fadeIn = function(element,item){

    jQuery(item.value).appendTo(element).fadeIn();

}

var changeText = function(element,item){

    jQuery(element).changeText(item.value, item.key);

}

var appendText = function(element,item){

    jQuery(element).appendText(item.value, item.key);

}

var removeClass = function(element,item){

    jQuery(element).removeClass(item.value);

}
var remove = function(element,item){

    jQuery(element).remove();

}
var clear = function(element,item){
    jQuery(element).empty();

}







function runResponse(item,i,self){

    var context;

    if(item.context !== "this"){

        context = item.context;

    } else {

        context = self;

    }
    methods[item.method](context,item);

}



jQuery(document).ready(function(){
    var pathname = window.location.pathname;
    jQuery('body').on('click','.click', function(e){
        e.preventDefault();
        var id = jQuery(this).attr('id');
        jQuery.ajax({url:pathname,data:{event:'click',uid:id},dataType:'json',context:this})    
            .done(function(result){
                var self = this;
                jQuery.each(result,function(i,item){
                    runResponse(item,i,self);
                });
            });
    });
    
    jQuery('body').on('click','.submit', function(e){
        e.preventDefault();
        var id = jQuery(this).attr('id');
        var form = jQuery(this).closest('form');
        
        jQuery.ajax({url:pathname,data:form.serialize()+'&event=submit&uid='+id,dataType:'json',context:this})    
            .done(function(result){
                var self = this;
                jQuery.each(result,function(i,item){
                    runResponse(item,i,self);
                });
            });
    });
    
    jQuery('body').on('click','.location', function(e){
        e.preventDefault();
       
        var id = jQuery(this).attr('id');
        if (!navigator.geolocation) {
            alert("Geolocation is not supported by this browser.");
            return false;
        }
        navigator.geolocation.getCurrentPosition(function(position){
            var latitude = position.coords.latitude;
            var longitude = position.coords.longitude;
            jQuery.ajax({url:pathname,data:{event:'location',uid:id,lat:latitude,long: longitude },dataType:'json',context:this})    
                .done(function(result){
                var self = this;
                jQuery.each(result,function(i,item){
                    runResponse(item,i,self);
                });
            });
        });
        
       
    });
});




