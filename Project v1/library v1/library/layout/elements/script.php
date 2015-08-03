<?php
/**
 * Description of script
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;
use app\request;



class script {
    private static $script = "";
    private static $response = array();
    private static $startScript = false;
    private static $listScript = array();
    private static $src = array("https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js","/js/bootstrap.min.js","/js/script.js");
    
    public static function start(){
        self::$startScript = true;
    }
    public static function isActive(){
        return self::$startScript;
    }

    public static function addSrc($src){
        self::$src[] = $src;
    }
    public static function getSrc(){
        $script = "";
        foreach(self::$src as $src){
            $script .= "<script src=\"$src\"></script>";
        }
        return $script;
    }
    
  
    public static function event(element $element,$event){
        if(!array_key_exists($event, self::$listScript)){
            self::bind($event);
        }
    }

    private static function bind($event){
        if($event == 'click'){
            self::$script = "jQuery('.click').$event(function(){%s});";
            self::$script = self::ajax($event);
            self::$listScript[$event] = self::$script;
        }
    }
    private static function ajax($event){
        $request = "var id = '#'+jQuery(this).attr('id');"
                .   "jQuery.ajax({url:'/".request::$uri["controller"]."/".request::$uri["action"]."',data:{event:'$event',uid:id},dataType:'json',context:this})";    
        $done = ".done(function(result){%s});";
        $callback = "var self = this;"
                . "jQuery.each(result,function(i,item){"
                .       "runResponse(item,i,self);" 
                . "});";
        $done = sprintf($done,$callback);
        $ajax = $request.$done;
        return sprintf(self::$script,$ajax);     
    }
    public static function loadMore($opts,$id){
        $data = array();
        foreach($opts as $key => $value){
            $data[] = $key.":'".$value."'";
        }
        $dt = implode(",",$data); 
        self::$listScript[] = "var is_loading = false;
                    jQuery(window).scroll(function() {
                    if(jQuery(window).scrollTop() + jQuery(window).height()  > jQuery(document).height() - jQuery('footer').height()) {
                        if (is_loading == false) {
                            is_loading = true;
                            var last_id = jQuery(\"#$id\").children('.response').eq(-1).attr('id');
                      
                            jQuery('.loader').show();
                            jQuery.ajax({
                                url: '/".request::$uri["controller"]."/ajax',
                                type: 'POST',
                                dataType:'html',
                                data: {last_id:last_id,$dt}
                            }).done(function(data){
                                if(data != 'false'){
                                    jQuery('#$id').append(data);
                                }
                            }).fail(function(jqXHR, textStatus, errorThrown) {
                                var alerta = jQuery('.alert');
                                if(typeof alerta !== 'undefined'){
                                    jQuery('.alert').remove();
                                }
                                jQuery('.loader').parent().prepend('<div class=\"alert alert-warning alert-dismissible\" role=\"alert\"><button type=\"button\" class=\"close\" data-dismiss=\"alert\" aria-label=\"Close\"><span aria-hidden=\"true\">&times;</span></button><strong>Algo deu errado </strong>'+jqXHR.responseText+'</div>');
                            }).always(function(jqXHR) {
                                jQuery('.loader').hide();
                                if(jqXHR != 'false'){
                                    is_loading = false;
                                }
                            });
                        }
                   }
                });";
    }

    public static function addValue($value,$context,$method){
        self::$response[] = array("context" => $context,"method" => $method, "value" => sprintf($value));
    }
    public static function addKeyValue($key,$value,$context,$method){
        self::$response[] = array("context" => $context,"method" => $method,"key" =>$key, "value" => $value);
    }
    public static function getScript(){
        $script = self::getSrc();
        $script .= "<script>jQuery(document).ready(function(){";
        foreach(self::$listScript as $jquery){
            $script .= $jquery;
        }
        $script .= " });</script>";
        return $script;
    }
    public static function getResponse(){
        return json_encode(self::$response);
    }
}