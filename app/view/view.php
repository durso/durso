<?php
namespace app\view;
use library\layout\elements\element;

class view
{
private $data = array();

private $files = array();

private $error = false;


private $elements;


public function assign($variable, $value)
{
    $this->data[$variable] = $value;

}
public function add(element $element){
    $this->elements[] = $element;
}

private function renderElements(){
    if(count($this->elements) > 0){
        foreach($this->elements as $element){
            print $element;
        }
    }
}
public function render()
{
    extract($this->data);
    require(TEMPLATE_PATH.DS."head.php");
    if($this->error){
        require(VIEW_PATH . DS . "error.php");
    }else {
        $this->renderElements();
    }
    require(TEMPLATE_PATH.DS."footer.php");
    
}
}
?>
